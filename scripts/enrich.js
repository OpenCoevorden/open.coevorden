import fs from "fs";
import crypto from "crypto";
import matter from "gray-matter";
import OpenAI from "openai";
import { globSync } from "glob";
import pLimit from "p-limit";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ------------------ CONFIG ------------------ */

const MODEL = process.env.AI_MODEL || "gpt-5.4-mini";
const MAX_CONCURRENT = parseInt(process.env.CONCURRENCY || "1", 10);
const MAX_TOKENS_PER_REQUEST = parseInt(process.env.MAX_TOKENS_PER_REQUEST || "50000", 10); // Veiligheidsmarge voor milestones
const MAX_SUMMARY_TOKENS = parseInt(process.env.MAX_SUMMARY_TOKENS || "30000", 10); // Harde bovengrens voor de samenvatting
const MAX_CHUNKS_PER_FILE = parseInt(process.env.MAX_CHUNKS_PER_FILE || "6", 10);
const REQUEST_DELAY_MS = parseInt(process.env.REQUEST_DELAY_MS || "1500", 10);
const DOC_YEAR = process.env.DOC_YEAR || "2023";
const DRY_RUN = process.env.DRY_RUN === "1";

/* --- EMBEDDINGS (semantische retrieval voor lange documenten) --- */
// Alleen documenten langer dan EMBED_THRESHOLD krijgen embeddings. Kortere
// documenten worden in de browser toch volledig meegestuurd, dus daar is
// retrieval overbodig. Stem EMBED_THRESHOLD af op FULL_LIMIT in single.html.
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const EMBED_DIM = parseInt(process.env.EMBED_DIM || "256", 10); // moet gelijk zijn aan de worker + single.html
const EMBED_THRESHOLD = parseInt(process.env.EMBED_THRESHOLD || "120000", 10);
const RETRIEVAL_CHUNK_CHARS = parseInt(process.env.RETRIEVAL_CHUNK_CHARS || "1500", 10);
const EMBED_BATCH = parseInt(process.env.EMBED_BATCH || "64", 10);

const SYSTEM_PROMPT =
  "Je bent een expert in Nederlandse Woo-dossiers. Taak: Extraheer een chronologische tijdlijn en samenvatting. " +
  "Neem alleen kerngebeurtenissen op: indiening aanvraag, besluit, bezwaar/beroep, uitspraak, verlenging, intrekking. " +
  "Laat proceduregebeurtenissen zoals ontvangstbevestigingen, interne herinneringen en correspondentie zonder inhoudelijke wijziging weg. " +
  "Gebruik ISO datums (YYYY-MM-DD). Negeer vóór 2020.";

/* ------------------ UTIL ------------------ */

const estimateTokens = (text) => Math.ceil(text.length / 4);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function hashContent(content) {
  return crypto.createHash("sha256").update(content).digest("hex").slice(0, 16);
}

// Lenient ISO-datum parser: accepteert ook niet-gepadde vormen zoals "2023-1-5"
function normalizeDate(dateStr) {
  if (typeof dateStr !== "string") return null;
  const m = dateStr.trim().match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (!m) return null;
  const year = parseInt(m[1], 10);
  const month = parseInt(m[2], 10);
  const day = parseInt(m[3], 10);
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function writeFrontmatter(file, content, data) {
  if (DRY_RUN) {
    console.log(`   (dry-run) zou schrijven: ${file}`);
    return;
  }
  // Atomic write: eerst naar tmp-bestand, dan renamen. Voorkomt corrupte
  // .md bestanden als het proces halverwege een write crasht.
  const tmpFile = `${file}.tmp`;
  fs.writeFileSync(tmpFile, matter.stringify(content, data));
  fs.renameSync(tmpFile, file);
}

/* ------------------ RETRY WRAPPER ------------------ */

async function withRetry(fn, label, retries = 5) {
  let lastErr;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const isRateLimit = err?.status === 429 || err?.message?.includes("Rate limit");
      const isContext = err?.status === 400 || err?.message?.includes("maximum context length");

      if (isRateLimit) {
        console.warn(`⏳ Rate limit geraakt (${label}). Wachten... (${i + 1}/${retries})`);
        await sleep(2000 * Math.pow(2, i));
        continue;
      }

      if (isContext) {
        console.warn(`⚠️  Context te groot (${label}), blok overgeslagen.`);
        return null;
      }
      throw err;
    }
  }
  throw lastErr;
}

/* ------------------ CHUNKING & FILTERING ------------------ */

// Alleen ondubbelzinnige boilerplate wordt weggegooid. Als een paragraaf óók
// een datum of een inhoudelijk keyword bevat, blijft hij staan — we willen
// nooit een echte gebeurtenis kwijtraken omdat er toevallig een standaardzin
// naast staat.
const IMPORTANT_RX =
  /\b(woo|besluit|verzoek|publicatie|termijn|afgehandeld|vastgesteld|toegekend|verlengd|ingetrokken|bezwaar|beroep|uitspraak|zienswijze)\b/i;

const DATE_RX =
  /(\d{4}-\d{2}-\d{2})|(\d{1,2}\s*(?:jan(?:uari)?|feb(?:ruari)?|mrt|maart|apr(?:il)?|mei|jun[i]?|jul[i]?|aug(?:ustus)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{4})/i;

const BOILERPLATE_RX = [
  /wettelijk kader.{0,20}artikel 5\./i,
  /aan deze brief kunnen geen rechten worden ontleend/i,
  /voor (algemene )?vragen kunt u (contact opnemen|bellen)/i,
  /^\s*ontvangstbevestiging\s*$/i,
];

function extractRelevantBlocks(text) {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
    .filter((p) => {
      const hasSignal = IMPORTANT_RX.test(p) || DATE_RX.test(p) || p.length > 80;
      const isPureBoilerplate =
        BOILERPLATE_RX.some((rx) => rx.test(p)) && !DATE_RX.test(p) && !IMPORTANT_RX.test(p);
      return hasSignal && !isPureBoilerplate;
    });
}

function buildSafeChunks(blocks) {
  // Blocks staan in originele documentvolgorde (chronologisch), dus chunks
  // dat ook. Dit voorkomt dat een datum en de bijbehorende gebeurtenis in
  // verschillende, ver uit elkaar verwerkte chunks belanden.
  const chunks = [];
  let current = [];
  let tokens = 0;

  for (const block of blocks) {
    const t = estimateTokens(block);

    // Eén blok is extreem groot (bijv. geen alinea-scheidingen in de PDF):
    // geforceerd in kleinere stukken knippen om crashes te voorkomen.
    if (t > MAX_TOKENS_PER_REQUEST) {
      let remainingText = block;
      const sliceSize = MAX_TOKENS_PER_REQUEST * 4;
      while (remainingText.length > 0) {
        chunks.push([remainingText.substring(0, sliceSize)]);
        remainingText = remainingText.substring(sliceSize);
      }
      continue;
    }

    if (tokens + t > MAX_TOKENS_PER_REQUEST) {
      if (current.length) chunks.push(current);
      current = [block];
      tokens = t;
    } else {
      current.push(block);
      tokens += t;
    }
  }
  if (current.length) chunks.push(current);
  return chunks;
}

function extractSummaryBlocksSmart(text) {
  const paragraphs = text.split(/\n\s*\n/);
  let summaryText = paragraphs
    .map((p) => {
      let score = 0;
      const lower = p.toLowerCase();
      if (/besluit|beslissing|toegekend|afgewezen|verlengd|gegrond|ongegrond/.test(lower)) score += 5;
      if (/aanvraag|verzoek|reactie|zienswijze|document|onderzoek|rapport/.test(lower)) score += 3;
      if (/college|burgemeester|gemeente|bestuurlijk|afdeling/.test(lower)) score += 2;
      if (p.length > 200) score += 1;
      return { text: p.trim(), score };
    })
    .filter((p) => p.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 30)
    .map((p) => p.text)
    .join("\n\n");

  if (estimateTokens(summaryText) > MAX_SUMMARY_TOKENS) {
    summaryText = summaryText.substring(0, MAX_SUMMARY_TOKENS * 4);
  }
  return summaryText;
}

/* ------------------ RETRIEVAL EMBEDDINGS ------------------ */

// Kleine stukjes speciaal voor semantische retrieval — losstaand van de grote
// milestone-chunks. Elk stukje wordt straks een vector waarmee de browser de
// vraag semantisch matcht (synoniemen/parafrases inbegrepen).
function chunkForRetrieval(text, size = RETRIEVAL_CHUNK_CHARS) {
  const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  let buf = "";
  for (const p of paras) {
    if (p.length > size) {
      // Paragraaf groter dan de chunkgrootte: hard opknippen.
      if (buf) { chunks.push(buf); buf = ""; }
      for (let i = 0; i < p.length; i += size) chunks.push(p.slice(i, i + size));
      continue;
    }
    if (buf && buf.length + p.length + 1 > size) { chunks.push(buf); buf = ""; }
    buf += (buf ? "\n" : "") + p;
  }
  if (buf) chunks.push(buf);
  return chunks;
}

async function embedChunks(chunks, label = "") {
  const vectors = [];
  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    const batch = chunks.slice(i, i + EMBED_BATCH);
    const res = await withRetry(
      () => openai.embeddings.create({ model: EMBED_MODEL, dimensions: EMBED_DIM, input: batch }),
      `${label} [embed ${i}]`
    );
    if (!res) return null; // context te groot / mislukt → hele blob afblazen
    for (const d of res.data) vectors.push(d.embedding);
  }
  return vectors;
}

// Vector → int8 → base64. Eén byte per dimensie i.p.v. ~7 tekens als JSON-float,
// zodat de frontmatter compact blijft.
function packVector(vec) {
  const buf = Buffer.alloc(vec.length);
  for (let i = 0; i < vec.length; i++) {
    buf[i] = Math.round(Math.max(-1, Math.min(1, vec[i])) * 127) & 0xff;
  }
  return buf.toString("base64");
}

// Bouwt het volledige retrieval-blok als één base64-string. Door alles in één
// string te stoppen voorkomen we YAML-escaping-problemen in de frontmatter.
async function buildRetrievalBlob(content, label) {
  const chunks = chunkForRetrieval(content);
  if (chunks.length === 0) return null;
  const vectors = await embedChunks(chunks, label);
  if (!vectors || vectors.length !== chunks.length) return null;
  const payload = { d: EMBED_DIM, c: chunks.map((t, i) => ({ t, v: packVector(vectors[i]) })) };
  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
}

/* ------------------ AI ------------------ */

async function analyzeContent(textBlocks, label = "") {
  const isEmpty =
    !textBlocks ||
    (Array.isArray(textBlocks) && textBlocks.length === 0) ||
    (typeof textBlocks === "string" && textBlocks.trim().length === 0);
  if (isEmpty) return null;

  return withRetry(async () => {
    const response = await openai.chat.completions.create({
      model: MODEL,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: `Geef STRICT JSON:\n{\n  "summary": "max 2 zinnen",\n  "milestones": [{ "date": "YYYY-MM-DD", "event": "kort" }]\n}\n\nTEKST:\n${
            Array.isArray(textBlocks) ? textBlocks.join("\n\n") : textBlocks
          }`,
        },
      ],
    });

    const raw = response.choices?.[0]?.message?.content;
    if (!raw) {
      console.warn(`⚠️  Lege AI-response (${label})`);
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (err) {
      console.warn(`⚠️  Ongeldige JSON van model (${label}): ${err.message}`);
      return null;
    }
  }, label);
}

/* ------------------ CLEANING ------------------ */

function cleanMilestones(milestones) {
  const seen = new Set();
  return milestones
    .map((m) => ({
      date: normalizeDate(m.date),
      event: typeof m.event === "string" ? m.event.trim() : String(m.event ?? "").trim(),
    }))
    .filter((m) => m.date && m.event.length > 0)
    .filter((m) => parseInt(m.date.slice(0, 4), 10) >= 2020)
    .filter((m) => {
      const id = `${m.date}-${m.event.toLowerCase()}`;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    })
    .sort((a, b) => a.date.localeCompare(b.date));
}

/* ------------------ FILE PROCESSING ------------------ */

async function processFile(file, stats) {
  let data, content;
  try {
    const raw = fs.readFileSync(file, "utf8");
    ({ data, content } = matter(raw));
  } catch (err) {
    console.error(`❌ Kan bestand niet lezen: ${file} | ${err.message}`);
    stats.failed++;
    return;
  }

  const contentHash = hashContent(content);

  // Heeft dit (lange) document nog embeddings nodig? Dan mag de skip-check het
  // niet overslaan, ook al is de AI-tekst al "done".
  const needsEmbedding = content.length > EMBED_THRESHOLD && data.ai_retrieval_hash !== contentHash;

  // Skip alleen als eerder succesvol verwerkt ÉN de content sindsdien niet
  // is gewijzigd ÉN er geen embeddings meer ontbreken.
  if (data.ai_status === "done" && data.ai_content_hash === contentHash && !needsEmbedding) {
    stats.skipped++;
    return;
  }

  try {
    const blocks = extractRelevantBlocks(content);

    if (blocks.length === 0) {
      console.warn(`⚠️  Geen relevante content gevonden, overgeslagen: ${file}`);
      data.ai_status = "no_content";
      data.ai_content_hash = contentHash;
      data.ai_processed_at = new Date().toISOString();
      writeFrontmatter(file, content, data);
      stats.noContent++;
      return;
    }

    // --- SUMMARY ---
    let summary = "";
    let summaryOk = false;
    try {
      const summaryInput = extractSummaryBlocksSmart(content);
      const summaryResult = await analyzeContent(summaryInput, `${file} [summary]`);
      summary = summaryResult?.summary?.trim() || "";
      summaryOk = summary.length > 0;
    } catch (err) {
      console.error(`❌ Summary mislukt voor ${file} | ${err.message}`);
    }

    // --- MILESTONES ---
    const chunks = buildSafeChunks(blocks);
    const chunksToProcess = chunks.slice(0, MAX_CHUNKS_PER_FILE);
    if (chunks.length > MAX_CHUNKS_PER_FILE) {
      console.warn(
        `⚠️  ${file}: ${chunks.length} chunks gevonden, slechts ${MAX_CHUNKS_PER_FILE} verwerkt ` +
          `(zet MAX_CHUNKS_PER_FILE hoger als dit document waarschijnlijk milestones mist).`
      );
    }

    let allMilestones = [];
    let milestonesOk = true;
    for (const [i, chunk] of chunksToProcess.entries()) {
      await sleep(REQUEST_DELAY_MS);
      try {
        const result = await analyzeContent(chunk, `${file} [chunk ${i + 1}/${chunksToProcess.length}]`);
        if (result?.milestones?.length) {
          allMilestones.push(...result.milestones);
        }
      } catch (err) {
        console.error(`❌ Milestone-chunk ${i + 1} mislukt voor ${file} | ${err.message}`);
        milestonesOk = false;
      }
    }

    const cleaned = cleanMilestones(allMilestones);

    data.summary = summary;
    data.milestones = cleaned;
    data.ai_content_hash = contentHash;
    data.ai_processed_at = new Date().toISOString();
    data.ai_status = summaryOk && milestonesOk ? "done" : "partial";

    // --- EMBEDDINGS (alleen lange documenten) ---
    // Losstaand van de summary/milestone-status: ook een 'partial' document mag
    // gewoon embeddings krijgen. We herberekenen alleen als de content wijzigde.
    if (content.length > EMBED_THRESHOLD && data.ai_retrieval_hash !== contentHash) {
      try {
        await sleep(REQUEST_DELAY_MS);
        const blob = await buildRetrievalBlob(content, `${file} [embed]`);
        if (blob) {
          data.ai_retrieval = blob;
          data.ai_retrieval_hash = contentHash;
          console.log(`   🔎 embeddings toegevoegd (${file})`);
        } else {
          console.warn(`⚠️  Embeddings overgeslagen voor ${file} (lege of mislukte blob)`);
        }
      } catch (err) {
        console.error(`❌ Embeddings mislukt voor ${file} | ${err.message}`);
      }
    } else if (content.length <= EMBED_THRESHOLD && data.ai_retrieval) {
      // Document is korter geworden en heeft geen retrieval meer nodig: opruimen.
      delete data.ai_retrieval;
      delete data.ai_retrieval_hash;
    }

    writeFrontmatter(file, content, data);

    if (data.ai_status === "done") {
      console.log(`✅ ${file} (${cleaned.length} milestones)`);
      stats.done++;
    } else {
      console.log(`🟡 ${file} deels verwerkt (${cleaned.length} milestones) — zie logs hierboven`);
      stats.partial++;
    }
  } catch (err) {
    console.error(`❌ Onverwachte fout bij ${file} | ${err.message}`);
    stats.failed++;
  }
}

/* ------------------ MAIN ------------------ */

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("❌ OPENAI_API_KEY ontbreekt. Zet deze environment variable en probeer opnieuw.");
    process.exit(1);
  }

  const files = globSync(`docs/${DOC_YEAR}/**/*.md`);
  if (files.length === 0) {
    console.warn(`⚠️  Geen .md bestanden gevonden voor docs/${DOC_YEAR}/**/*.md — klopt DOC_YEAR ("${DOC_YEAR}")?`);
    return;
  }

  console.log(
    `📂 ${files.length} bestand(en) gevonden voor jaar ${DOC_YEAR}${DRY_RUN ? " (DRY RUN, er wordt niets weggeschreven)" : ""}`
  );

  const stats = { done: 0, partial: 0, noContent: 0, skipped: 0, failed: 0 };
  const limit = pLimit(MAX_CONCURRENT);

  const results = await Promise.allSettled(files.map((f) => limit(() => processFile(f, stats))));
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`❌ Onverwachte fout bij ${files[i]}: ${r.reason?.message || r.reason}`);
      stats.failed++;
    }
  });

  console.log("\n──────── Samenvatting ────────");
  console.log(`✅ Volledig verwerkt : ${stats.done}`);
  console.log(`🟡 Deels verwerkt    : ${stats.partial}`);
  console.log(`⚪ Geen content      : ${stats.noContent}`);
  console.log(`⏭️  Overgeslagen      : ${stats.skipped}`);
  console.log(`❌ Mislukt           : ${stats.failed}`);
  console.log("───────────────────────────────");

  if (stats.partial > 0 || stats.failed > 0) {
    console.log("Tip: draai het script opnieuw — bestanden met status 'partial' of een leesfout worden automatisch herpakt.");
  }
}

main().catch((err) => {
  console.error("❌ Fatale fout:", err);
  process.exit(1);
});
