/**
 * STAGELINE — Tour Attendance Predictor
 *
 * Data flow:
 *   artists.json  → artist select → auto-fill gender/generation/members/company/debut
 *   venues.json   → continent → country → venue selects → auto-fill type/capacity/city
 *   show date     → compute years_since_debut + artist_age_bucket
 *   show_nights   → stepper (1–6)
 *   All fields ready → POST /predict/ → display result
 */

const API_BASE   = "http://localhost:8000";
const STATIC_DIR = "../data/static";       // path relative to index.html

// ── Age bucket thresholds (must match training logic) ──────────────────────
const AGE_BUCKETS = [
  { max: 3,        label: "rookie"      },
  { max: 6,        label: "growing"     },
  { max: 12,       label: "established" },
  { max: Infinity, label: "legacy"      },
];

function getAgeBucket(yearsSinceDebut) {
  return AGE_BUCKETS.find(b => yearsSinceDebut <= b.max).label;
}

function getYearsSinceDebut(debutDateStr, showDateStr) {
  const debut = new Date(debutDateStr);
  const show  = new Date(showDateStr);
  const diff  = (show - debut) / (1000 * 60 * 60 * 24 * 365.25);
  return Math.max(0, Math.floor(diff));
}

// ── State ──────────────────────────────────────────────────────────────────
let artists = [];
let venues  = [];
let selectedArtist = null;
let selectedVenue  = null;

// ── DOM refs ───────────────────────────────────────────────────────────────
const artistSelect    = document.getElementById("artist_name");
const continentSelect = document.getElementById("continent");
const countrySelect   = document.getElementById("country");
const venueSelect     = document.getElementById("venue_name");
const showDateInput   = document.getElementById("show_date");
const showNightsInput = document.getElementById("show_nights");
const nightsDisplay   = document.getElementById("nights-display");
const nightsDec       = document.getElementById("nights-dec");
const nightsInc       = document.getElementById("nights-inc");
const submitBtn       = document.getElementById("submit-btn");
const form            = document.getElementById("prediction-form");
const resultSection   = document.getElementById("result-section");
const resultNumber    = document.getElementById("result-number");
const resultContext   = document.getElementById("result-context");
const resetBtn        = document.getElementById("reset-btn");
const footerModel     = document.getElementById("footer-model");

// ── Helpers ────────────────────────────────────────────────────────────────
function populate(select, options, labelFn, valueFn) {
  const first = select.options[0];           // keep placeholder
  select.innerHTML = "";
  select.appendChild(first);
  options.forEach(o => {
    const opt = document.createElement("option");
    opt.value       = valueFn(o);
    opt.textContent = labelFn(o);
    select.appendChild(opt);
  });
}

function setPill(id, text) {
  document.getElementById(id).textContent = text || "";
}

function showMeta(id) {
  const el = document.getElementById(id);
  el.classList.add("visible");
  el.removeAttribute("aria-hidden");
}

function hideMeta(id) {
  const el = document.getElementById(id);
  el.classList.remove("visible");
  el.setAttribute("aria-hidden", "true");
}

function formatNumber(n) {
  return new Intl.NumberFormat("en-US").format(n);
}

function checkFormReady() {
  const ready =
    selectedArtist &&
    selectedVenue  &&
    showDateInput.value &&
    showNightsInput.value;
  submitBtn.disabled = !ready;
}

// ── Load static data ────────────────────────────────────────────────────────
async function loadData() {
  const [artistsRes, venuesRes] = await Promise.all([
    fetch(`${STATIC_DIR}/artists.json`),
    fetch(`${STATIC_DIR}/venues.json`),
  ]);

  artists = await artistsRes.json();
  venues  = await venuesRes.json();

  // Populate artist select
  populate(
    artistSelect,
    artists,
    a => a.artist_name,
    a => a.id,
  );

  // Populate continent select (unique, sorted)
  const continents = [...new Set(venues.map(v => v.continent))].sort();
  populate(
    continentSelect,
    continents,
    c => c,
    c => c,
  );
}

// ── Artist selection ────────────────────────────────────────────────────────
artistSelect.addEventListener("change", () => {
  const id = parseInt(artistSelect.value);
  selectedArtist = artists.find(a => a.id === id) || null;

  if (!selectedArtist) {
    hideMeta("artist-meta");
    checkFormReady();
    return;
  }

  setPill("meta-gender",     selectedArtist.gender.toUpperCase());
  setPill("meta-members",    `${selectedArtist.members} members`);
  setPill("meta-generation", `GEN ${selectedArtist.generation}`);
  setPill("meta-company",    selectedArtist.company);
  setPill("meta-debut",      `debut ${selectedArtist.debut_date}`);
  showMeta("artist-meta");

  // Recompute date-derived fields if date is already set
  if (showDateInput.value) updateDateMeta();

  checkFormReady();
});

// ── Continent → Country ─────────────────────────────────────────────────────
continentSelect.addEventListener("change", () => {
  const continent = continentSelect.value;

  // Reset downstream
  selectedVenue = null;
  venueSelect.innerHTML = "<option value=''>— select country first —</option>";
  venueSelect.disabled  = true;
  hideMeta("venue-meta");

  if (!continent) {
    countrySelect.innerHTML = "<option value=''>— select continent first —</option>";
    countrySelect.disabled  = true;
    checkFormReady();
    return;
  }

  const countries = [...new Set(
    venues.filter(v => v.continent === continent).map(v => v.country)
  )].sort();

  countrySelect.innerHTML = "<option value=''>— select —</option>";
  countries.forEach(c => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = c;
    countrySelect.appendChild(opt);
  });
  countrySelect.disabled = false;

  checkFormReady();
});

// ── Country → Venue ─────────────────────────────────────────────────────────
countrySelect.addEventListener("change", () => {
  const country   = countrySelect.value;
  const continent = continentSelect.value;

  selectedVenue = null;
  hideMeta("venue-meta");

  if (!country) {
    venueSelect.innerHTML = "<option value=''>— select country first —</option>";
    venueSelect.disabled  = true;
    checkFormReady();
    return;
  }

  const filtered = venues.filter(
    v => v.continent === continent && v.country === country
  );

  venueSelect.innerHTML = "<option value=''>— select —</option>";
  filtered.forEach(v => {
    const opt = document.createElement("option");
    opt.value       = v.id;
    opt.textContent = v.venue_name;
    venueSelect.appendChild(opt);
  });
  venueSelect.disabled = false;

  checkFormReady();
});

// ── Venue selection ─────────────────────────────────────────────────────────
venueSelect.addEventListener("change", () => {
  const id = parseInt(venueSelect.value);
  selectedVenue = venues.find(v => v.id === id) || null;

  if (!selectedVenue) {
    hideMeta("venue-meta");
    checkFormReady();
    return;
  }

  setPill("meta-venue-type", selectedVenue.venue_type.toUpperCase());
  setPill("meta-capacity",   `${formatNumber(selectedVenue.venue_capacity)} cap.`);
  setPill("meta-city",       selectedVenue.city);
  showMeta("venue-meta");

  checkFormReady();
});

// ── Date input ──────────────────────────────────────────────────────────────
showDateInput.addEventListener("change", () => {
  updateDateMeta();
  checkFormReady();
});

function updateDateMeta() {
  if (!showDateInput.value || !selectedArtist) {
    hideMeta("date-meta");
    return;
  }

  const years  = getYearsSinceDebut(selectedArtist.debut_date, showDateInput.value);
  const bucket = getAgeBucket(years);

  setPill("meta-years-debut", `${years} yr since debut`);
  setPill("meta-age-bucket",  bucket.toUpperCase());
  showMeta("date-meta");
}

// ── Nights stepper ──────────────────────────────────────────────────────────
let nights = 1;

function setNights(n) {
  nights = Math.min(6, Math.max(1, n));
  nightsDisplay.textContent = nights;
  showNightsInput.value     = nights;
  nightsDec.disabled = nights === 1;
  nightsInc.disabled = nights === 6;
}

nightsDec.addEventListener("click", () => setNights(nights - 1));
nightsInc.addEventListener("click", () => setNights(nights + 1));

// ── Form submit ─────────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedArtist || !selectedVenue) return;

  const years  = getYearsSinceDebut(selectedArtist.debut_date, showDateInput.value);
  const bucket = getAgeBucket(years);

  const payload = {
    venue_name:        selectedVenue.venue_name,
    venue_type:        selectedVenue.venue_type,
    venue_capacity:    selectedVenue.venue_capacity,
    continent:         selectedVenue.continent,
    country:           selectedVenue.country,
    city:              selectedVenue.city,
    artist_name:       selectedArtist.artist_name,
    gender:            selectedArtist.gender,
    generation:        selectedArtist.generation,
    members:           selectedArtist.members,
    company:           selectedArtist.company,
    years_since_debut: years,
    artist_age_bucket: bucket,
    show_nights:       nights,
  };

  submitBtn.classList.add("loading");
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    resultNumber.textContent = formatNumber(data.predicted_attendance);
    resultContext.textContent =
      `${selectedArtist.artist_name} · ${selectedVenue.venue_name} · ${nights} night${nights > 1 ? "s" : ""}`;

    form.hidden           = true;
    resultSection.hidden  = false;

  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
    submitBtn.disabled = false;
  } finally {
    submitBtn.classList.remove("loading");
  }
});

// ── Reset ───────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  form.reset();
  form.hidden          = false;
  resultSection.hidden = true;

  selectedArtist = null;
  selectedVenue  = null;
  setNights(1);

  countrySelect.innerHTML = "<option value=''>— select continent first —</option>";
  countrySelect.disabled  = true;
  venueSelect.innerHTML   = "<option value=''>— select country first —</option>";
  venueSelect.disabled    = true;

  hideMeta("artist-meta");
  hideMeta("venue-meta");
  hideMeta("date-meta");

  submitBtn.disabled = true;
});

// ── Init ────────────────────────────────────────────────────────────────────
loadData().catch(err => {
  console.error("Failed to load static data:", err);
  alert("Could not load artist/venue data. Make sure the static JSON files are served correctly.");
});