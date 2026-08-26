// =============================================================================
// woo-publicaties-verrijken.js — AI-verrijking van de Woo-markdownbestanden
// =============================================================================
//
// WAT DOET DIT SCRIPT?
//   Loopt door de markdownbestanden onder onderwerpen/<DOC_PATH>/ (aangemaakt
//   door de sync-workflow) en laat OpenAI per dossier een korte samenvatting +
//   een chronologische tijdlijn (milestones) genereren. Die worden als extra
//   frontmatter (summary, milestones, ai_*) teruggeschreven, zodat single.html
//   ze kan tonen. Wordt aangeroepen door woo-publicaties_verrijken.yml.
//
//   Idempotent: een bestand wordt overgeslagen als ai_status op done staat.
//   Draai gerust opnieuw; "partial"/mislukte bestanden worden automatisch
//   herpakt.
//
// -----------------------------------------------------------------------------
// EXTERNE LIBRARIES / DEPENDENCIES (installeren via de workflow met:
//   npm install gray-matter openai glob fs-extra p-limit)
// -----------------------------------------------------------------------------
import fs from "fs";                 // Node-ingebouwd: bestanden lezen/schrijven.
import matter from "gray-matter";    // npm 'gray-matter': frontmatter parsen/schrijven.
import OpenAI from "openai";         // npm 'openai': officiële OpenAI-client.
import { globSync } from "glob";     // npm 'glob': .md-bestanden vinden via patroon.
import pLimit from "p-limit";        // npm 'p-limit': parallelle taken begrenzen.
// Let op: 'fs-extra' wordt door de workflow meegeïnstalleerd maar hier niet
// geïmporteerd; dat is onschadelijk. Updaten van libs: pas de npm install-regel
// in woo-publicaties_verrijken.yml aan én, indien nodig, deze imports.

/* ------------------ CONFIG ------------------
   Alle instelbare waarden staan hier. Vrijwel elke waarde is via een environment
   variable te overschrijven (handig om in de workflow te tunen zonder code te
   wijzigen); tussen haakjes staat telkens de default. */

// Het OpenAI-model voor samenvatting + milestones.
// GPT-5.6 Luna is de goedkoopste route voor extractie/samenvatting en heeft een
// context van ~1M tokens. Updaten: zet env AI_MODEL, of wijzig de default hier.
const MODEL = process.env.AI_MODEL || "gpt-5.6-luna";

// Verwerkingsmodus. "flex" is ongeveer de helft van het standaardtarief in ruil
// voor variabele (soms véél langere) latency — prima voor deze batchjob, die
// geen enkele latency-eis heeft. Zet SERVICE_TIER="" (leeg) voor het normale
// tarief, bijvoorbeeld tijdens een snelle test.
const SERVICE_TIER = process.env.SERVICE_TIER ?? "flex";

// Sommige nieuwere modellen accepteren geen 'temperature' meer in Chat
// Completions en antwoorden dan met een 400. Zet USE_TEMPERATURE=0 als het
// gekozen model daarover valt; de parameter wordt dan simpelweg weggelaten.
const USE_TEMPERATURE = process.env.USE_TEMPERATURE === "1";

// Harde time-out (ms) per OpenAI-call op clientniveau. Ruim gezet omdat
// flex-requests lang kunnen wachten op capaciteit.
const OPENAI_TIMEOUT_MS = parseInt(process.env.OPENAI_TIMEOUT_MS || "600000", 10);

// Aantal bestanden dat tegelijk verwerkt wordt (parallellisme).
// Updaten: zet env CONCURRENCY hoger voor snelheid; hoger = meer kans op rate limits.
const MAX_CONCURRENT = parseInt(process.env.CONCURRENCY || "1", 10);

// Max tokens per AI-request (bepaalt ook de chunk-grootte). Met een
// contextvenster van ~1M is dit geen echte beperking meer; ruim gezet zodat de
// meeste dossiers in één call passen en een datum nooit los raakt van de
// bijbehorende gebeurtenis. Let op: estimateTokens is een grove schatting
// (tekens/4), dus hou marge tot het echte venster.
// Updaten: env MAX_TOKENS_PER_REQUEST.
const MAX_TOKENS_PER_REQUEST = parseInt(process.env.MAX_TOKENS_PER_REQUEST || "60000", 10);

// Harde bovengrens op de tekst die naar de samenvatting-call gaat.
// Updaten: env MAX_SUMMARY_TOKENS. Voorkomt te grote samenvattings-requests.
const MAX_SUMMARY_TOKENS = parseInt(process.env.MAX_SUMMARY_TOKENS || "30000", 10);

// Max aantal chunks per bestand dat voor milestones wordt verwerkt.
// Kan laag omdat MAX_TOKENS_PER_REQUEST hierboven ruim staat.
// Updaten: env MAX_CHUNKS_PER_FILE hoger als lange dossiers milestones missen.
const MAX_CHUNKS_PER_FILE = parseInt(process.env.MAX_CHUNKS_PER_FILE || "6", 10);

// Pauze (ms) tussen AI-calls om rate limits te ontzien.
// Updaten: env REQUEST_DELAY_MS. Hoger = trager maar veiliger.
const REQUEST_DELAY_MS = parseInt(process.env.REQUEST_DELAY_MS || "1500", 10);

// Ondergrens voor milestone-jaartallen. Staat op ÉÉN plek: hij wordt zowel in
// de system-prompt geïnterpoleerd als in cleanMilestones() afgedwongen, zodat
// instructie en filter nooit uit elkaar kunnen lopen.
// Updaten: env MIN_YEAR.
const MIN_YEAR = parseInt(process.env.MIN_YEAR || "2010", 10);

// Pad-fragment onder onderwerpen/ dat verrijkt wordt, bv. "woo-publicaties/2023".
// MOET dezelfde casing hebben als in de workflows. Updaten: env DOC_PATH
// (gezet door woo-publicaties_verrijken.yml).
const DOC_PATH = process.env.DOC_PATH || "woo-publicaties/2023";

// Testmodus: bij "1" wordt er niets weggeschreven (alleen loggen).
// LET OP: dit blokkeert alleen het SCHRIJVEN — de OpenAI-calls worden wél
// gedaan en kosten dus gewoon tokens.
// Updaten: zet env DRY_RUN=1 om een proefrun te doen.
const DRY_RUN = process.env.DRY_RUN === "1";

// Instructie aan het model. Bepaalt wat wel/niet als milestone telt.
// Updaten: pas de regels aan; houd "gebruik ISO-datums" en de scope-afbakening intact.
const SYSTEM_PROMPT =
  "Je bent een expert in Nederlandse Woo-dossiers. Taak: Extraheer een chronologische tijdlijn en samenvatting. " +
  "Neem alleen kerngebeurtenissen op: indiening aanvraag, besluit, bezwaar/beroep, uitspraak, verlenging, intrekking. " +
  "Laat proceduregebeurtenissen zoals ontvangstbevestigingen, interne herinneringen en correspondentie zonder inhoudelijke wijziging weg. " +
  `Gebruik ISO datums (YYYY-MM-DD). Negeer vóór ${MIN_YEAR}.`;

// OpenAI-client. Leest de sleutel uit de omgeving (nooit hardcoden).
// Updaten: zet OPENAI_API_KEY als secret in de workflow.
//
// maxRetries: 0 — de client retryt standaard zelf. Samen met withRetry()
// hieronder zou je geneste exponentiële backoff krijgen en bij rate limits
// onnodig lang wachten. De retry-logica houden we bewust op één plek.
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: OPENAI_TIMEOUT_MS,
  maxRetries: 0,
});

/* ------------------ UTIL ------------------ */

const estimateTokens = (text) => Math.ceil(text.length / 4);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

      // 429 = rate limit. Bij service_tier "flex" kan 429 óók "capaciteit
      // tijdelijk niet beschikbaar" betekenen; dezelfde backoff werkt daarvoor.
      const isRateLimit = err?.status === 429 || err?.message?.includes("Rate limit");

      // ALLEEN een echte context-overschrijding overslaan — niet elke 400.
      // Een 400 door bijvoorbeeld een niet-ondersteunde parameter zou anders
      // stil worden weggeslikt als "context te groot", waarna het bestand
      // ten onrechte ai_status "done" krijgt en door de idempotentie-check
      // nooit meer wordt herpakt.
      const isContext =
        err?.code === "context_length_exceeded" ||
        err?.message?.includes("maximum context length");

      if (isRateLimit) {
        console.warn(`⏳ Rate limit / geen capaciteit (${label}). Wachten... (${i + 1}/${retries})`);
        await sleep(2000 * Math.pow(2, i));
        continue;
      }

      if (isContext) {
        console.warn(`⚠️  Context te groot (${label}), blok overgeslagen.`);
        return null;
      }

      // Alle overige fouten (o.a. 400 op een verkeerde parameter) gooien we
      // door, zodat ze zichtbaar worden en het bestand "partial" wordt.
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

// Lengte-drempel voor de vangnet-tak: een alinea ZONDER keyword en ZONDER
// datum haalt het alleen nog als hij substantieel is. Verlagen als er te veel
// bestanden op "no_content" uitkomen (bv. bij matige OCR of PDF's waar
// pdftotext de alinea's slecht scheidt).
const MIN_BLOCK_CHARS = parseInt(process.env.MIN_BLOCK_CHARS || "400", 10);

function extractRelevantBlocks(text) {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
    .filter((p) => {
      const hasSignal = IMPORTANT_RX.test(p) || DATE_RX.test(p) || p.length > MIN_BLOCK_CHARS;
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
      // Bewust GEEN score op institutionele termen (college/burgemeester/
      // gemeente/bestuurlijk/afdeling): die staan in vrijwel elke alinea van
      // een gemeentelijke brief en verhoogden de hele set gelijkmatig, zonder
      // onderscheidend vermogen. Let op het neveneffect: alinea's die alleen
      // hierop scoorden vallen nu op 0 en worden door de filter hieronder
      // weggegooid — de summary-input kan dus minder dan 30 alinea's bevatten.
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
      // Optionele parameters: alleen meesturen als ze gezet zijn, zodat een
      // model dat ze niet ondersteunt geen 400 teruggeeft.
      ...(USE_TEMPERATURE ? { temperature: 0 } : {}),
      ...(SERVICE_TIER ? { service_tier: SERVICE_TIER } : {}),
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
    .filter((m) => parseInt(m.date.slice(0, 4), 10) >= MIN_YEAR)
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

  console.log(`🔍 ${file}`);
  console.log(`   ai_status:       ${data.ai_status}`);

  // Sla bestanden over die succesvol zijn verrijkt.
  // Bestanden met "partial", "no_content" of zonder ai_status worden opnieuw verwerkt.
  if (data.ai_status === "done") {
    console.log(`⏭️  Overgeslagen: ${file}`);
    stats.skipped++;
    return;
  }

  try {
    const blocks = extractRelevantBlocks(content);

    if (blocks.length === 0) {
      console.warn(`⚠️  Geen relevante content gevonden, overgeslagen: ${file}`);
      data.ai_status = "no_content";
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
    data.ai_processed_at = new Date().toISOString();
    data.ai_status = summaryOk && milestonesOk ? "done" : "partial";

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

  const files = globSync(`onderwerpen/${DOC_PATH}/**/*.md`);
  if (files.length === 0) {
    console.warn(`⚠️  Geen .md bestanden gevonden voor onderwerpen/${DOC_PATH}/**/*.md — klopt DOC_PATH ("${DOC_PATH}")?`);
    return;
  }

  console.log(
    `📂 ${files.length} bestand(en) gevonden voor ${DOC_PATH}${DRY_RUN ? " (DRY RUN, er wordt niets weggeschreven)" : ""}`
  );
  console.log(
    `⚙️  model=${MODEL} tier=${SERVICE_TIER || "standaard"} temperature=${USE_TEMPERATURE ? "0" : "uit"} min_year=${MIN_YEAR}`
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
