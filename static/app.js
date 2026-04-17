const STAR_COUNT = 5;

// ===== State =====
const selectedMovies = [];
let searchTimeout = null;
let currentUser = null;
let authMode = "login";
let lastRecommendations = [];
let ratingQueue = [];
let currentRatingContext = null;
let currentRatingValue = 0;

const activeFilters = {
    language: "",
    genres: [],
};

// ===== DOM Elements =====
const searchInput = document.getElementById("searchInput");
const searchResults = document.getElementById("searchResults");
const selectedMoviesEl = document.getElementById("selectedMovies");
const selectedCountEl = document.getElementById("selectedCount");
const recommendBtn = document.getElementById("recommendBtn");
const recommendationsSection = document.getElementById("recommendationsSection");
const recommendationsEl = document.getElementById("recommendations");
const loadingEl = document.getElementById("loading");
const weightInfoEl = document.getElementById("weightInfo");

// Auth
const authLoggedOut = document.getElementById("authLoggedOut");
const authLoggedIn = document.getElementById("authLoggedIn");
const authUsername = document.getElementById("authUsername");
const showLoginBtn = document.getElementById("showLoginBtn");
const showRegisterBtn = document.getElementById("showRegisterBtn");
const logoutBtn = document.getElementById("logoutBtn");
const showHistoryBtn = document.getElementById("showHistoryBtn");

// Auth modal
const authModal = document.getElementById("authModal");
const authModalClose = document.getElementById("authModalClose");
const authModalTitle = document.getElementById("authModalTitle");
const authUsernameInput = document.getElementById("authUsernameInput");
const authPasswordInput = document.getElementById("authPasswordInput");
const authError = document.getElementById("authError");
const authSubmitBtn = document.getElementById("authSubmitBtn");
const authSwitchText = document.getElementById("authSwitchText");

// History modal
const historyModal = document.getElementById("historyModal");
const historyModalClose = document.getElementById("historyModalClose");
const historyCount = document.getElementById("historyCount");
const historyList = document.getElementById("historyList");

// Rating modal
const ratingModal = document.getElementById("ratingModal");
const ratingModalClose = document.getElementById("ratingModalClose");
const ratingMovieTitle = document.getElementById("ratingMovieTitle");
const starRatingEl = document.getElementById("starRating");
const ratingValueEl = document.getElementById("ratingValue");
const ratingSubmitBtn = document.getElementById("ratingSubmitBtn");
const ratingSkipBtn = document.getElementById("ratingSkipBtn");

// Filters
const toggleFiltersBtn = document.getElementById("toggleFiltersBtn");
const filtersPanel = document.getElementById("filtersPanel");
const languageFilter = document.getElementById("languageFilter");
const genreFilterList = document.getElementById("genreFilterList");
const applyFiltersBtn = document.getElementById("applyFiltersBtn");
const clearFiltersBtn = document.getElementById("clearFiltersBtn");
const activeFiltersSummary = document.getElementById("activeFiltersSummary");

// ===== Init =====
(async function initializeApp() {
    bindAuthEvents();
    bindHistoryEvents();
    bindSearchEvents();
    bindFilterEvents();
    bindRatingEvents();

    await Promise.all([checkSession(), loadFilterOptions()]);
    updateAuthUI();
    updateFilterSummary();
    updateSelectedMovies();
})();

// ===== Auth =====
async function checkSession() {
    try {
        const res = await fetch("/api/me");
        const data = await res.json();
        if (data.logged_in) {
            currentUser = data.username;
        }
    } catch (err) {
        currentUser = null;
    }
}

function bindAuthEvents() {
    showLoginBtn.addEventListener("click", () => openAuthModal("login"));
    showRegisterBtn.addEventListener("click", () => openAuthModal("register"));
    authModalClose.addEventListener("click", () => {
        authModal.style.display = "none";
    });
    authModal.addEventListener("click", (event) => {
        if (event.target === authModal) {
            authModal.style.display = "none";
        }
    });

    authSubmitBtn.addEventListener("click", submitAuth);
    authPasswordInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            submitAuth();
        }
    });

    logoutBtn.addEventListener("click", async () => {
        try {
            await fetch("/api/logout", { method: "POST" });
        } catch (err) {
            // best effort logout
        }

        currentUser = null;
        updateAuthUI();
    });
}

function updateAuthUI() {
    if (currentUser) {
        authLoggedOut.style.display = "none";
        authLoggedIn.style.display = "flex";
        authUsername.textContent = currentUser;
    } else {
        authLoggedOut.style.display = "flex";
        authLoggedIn.style.display = "none";
        authUsername.textContent = "";
    }

    if (lastRecommendations.length > 0) {
        renderRecommendations(lastRecommendations);
    }
}

function openAuthModal(mode) {
    authMode = mode;
    authModal.style.display = "flex";
    authError.style.display = "none";
    authUsernameInput.value = "";
    authPasswordInput.value = "";

    if (mode === "login") {
        authModalTitle.textContent = "Log In";
        authSubmitBtn.textContent = "Log In";
        authSwitchText.innerHTML = 'Don\'t have an account? <a id="switchToRegister">Register</a>';
        authSwitchText.querySelector("a").addEventListener("click", () => openAuthModal("register"));
    } else {
        authModalTitle.textContent = "Register";
        authSubmitBtn.textContent = "Create Account";
        authSwitchText.innerHTML = 'Already have an account? <a id="switchToLogin">Log In</a>';
        authSwitchText.querySelector("a").addEventListener("click", () => openAuthModal("login"));
    }

    authUsernameInput.focus();
}

async function submitAuth() {
    const username = authUsernameInput.value.trim();
    const password = authPasswordInput.value;

    if (!username || !password) {
        showAuthError("Please enter username and password.");
        return;
    }

    try {
        const endpoint = authMode === "login" ? "/api/login" : "/api/register";
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();

        if (!res.ok) {
            showAuthError(data.error || "Authentication failed.");
            return;
        }

        currentUser = data.username;
        authModal.style.display = "none";
        updateAuthUI();
    } catch (err) {
        showAuthError("Network error. Please try again.");
    }
}

function showAuthError(message) {
    authError.textContent = message;
    authError.style.display = "block";
}

// ===== History =====
function bindHistoryEvents() {
    showHistoryBtn.addEventListener("click", openHistoryModal);
    historyModalClose.addEventListener("click", () => {
        historyModal.style.display = "none";
    });
    historyModal.addEventListener("click", (event) => {
        if (event.target === historyModal) {
            historyModal.style.display = "none";
        }
    });
}

async function openHistoryModal() {
    historyModal.style.display = "flex";
    historyList.innerHTML = '<p class="history-empty">Loading...</p>';
    historyCount.textContent = "";

    try {
        const res = await fetch("/api/history");
        const data = await res.json();

        if (!res.ok) {
            historyList.innerHTML = `<p class="history-empty">${escapeHtml(data.error || "Failed to load history.")}</p>`;
            return;
        }

        renderHistory(data.history || []);
    } catch (err) {
        historyList.innerHTML = '<p class="history-empty">Failed to load history.</p>';
    }
}

function renderHistory(history) {
    historyCount.textContent = `${history.length} movie${history.length !== 1 ? "s" : ""} watched`;

    if (history.length === 0) {
        historyList.innerHTML = '<p class="history-empty">No movies in your watch history yet.</p>';
        return;
    }

    historyList.innerHTML = history.map((item) => {
        const watchedOn = new Date(item.timestamp * 1000).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
        });

        return `
            <div class="history-item">
                <div class="history-item-info">
                    <div class="history-item-title">${escapeHtml(item.title)} (${item.release_year})</div>
                    <div class="history-item-meta">Watched: ${watchedOn}</div>
                    <div class="history-rating-row">
                        <div class="inline-stars">
                            ${buildMiniStars(item.index, item.rating || 0)}
                        </div>
                        <span class="history-item-rating">
                            ${item.rating ? `${item.rating}/${STAR_COUNT} stars` : "Not rated yet"}
                        </span>
                    </div>
                </div>
                <button class="remove-btn history-remove-btn" data-index="${item.index}" title="Remove from history">&times;</button>
            </div>
        `;
    }).join("");

    historyList.querySelectorAll(".history-remove-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            const movieIndex = Number(button.dataset.index);
            try {
                const res = await fetch(`/api/history/${movieIndex}`, { method: "DELETE" });
                if (res.ok) {
                    await openHistoryModal();
                }
            } catch (err) {
                // no-op
            }
        });
    });

    historyList.querySelectorAll(".mini-star").forEach((star) => {
        star.addEventListener("click", async () => {
            const movieIndex = Number(star.dataset.index);
            const rating = Number(star.dataset.value);
            const updated = await saveMovieRating(movieIndex, rating);
            if (updated) {
                await openHistoryModal();
            }
        });
    });
}

function buildMiniStars(movieIndex, currentRating) {
    let stars = "";
    for (let value = 1; value <= STAR_COUNT; value += 1) {
        const activeClass = value <= currentRating ? "active" : "";
        stars += `<span class="mini-star ${activeClass}" data-index="${movieIndex}" data-value="${value}">&#9733;</span>`;
    }
    return stars;
}

// ===== Filters =====
function bindFilterEvents() {
    toggleFiltersBtn.addEventListener("click", () => {
        const shouldOpen = filtersPanel.style.display === "none";
        filtersPanel.style.display = shouldOpen ? "block" : "none";
    });

    applyFiltersBtn.addEventListener("click", () => {
        activeFilters.language = languageFilter.value;
        activeFilters.genres = Array.from(
            genreFilterList.querySelectorAll("input:checked")
        ).map((input) => input.value);

        updateFilterSummary();
        filtersPanel.style.display = "none";

        if (searchInput.value.trim().length >= 2) {
            fetchSearchResults(searchInput.value.trim());
        }
    });

    clearFiltersBtn.addEventListener("click", () => {
        languageFilter.value = "";
        genreFilterList.querySelectorAll("input").forEach((input) => {
            input.checked = false;
        });

        activeFilters.language = "";
        activeFilters.genres = [];
        updateFilterSummary();
        filtersPanel.style.display = "none";

        if (searchInput.value.trim().length >= 2) {
            fetchSearchResults(searchInput.value.trim());
        }
    });
}

async function loadFilterOptions() {
    try {
        const [languageRes, genreRes] = await Promise.all([
            fetch("/api/languages"),
            fetch("/api/genres"),
        ]);
        const [languageData, genreData] = await Promise.all([
            languageRes.json(),
            genreRes.json(),
        ]);

        populateLanguageOptions(languageData.languages || []);
        populateGenreOptions(genreData.genres || []);
    } catch (err) {
        console.error("Failed to load filters:", err);
    }
}

function populateLanguageOptions(languages) {
    languageFilter.innerHTML = '<option value="">All languages</option>';

    languages.forEach((language) => {
        const option = document.createElement("option");
        option.value = language.code;
        option.textContent = `${language.label} (${language.count})`;
        languageFilter.appendChild(option);
    });
}

function populateGenreOptions(genres) {
    genreFilterList.innerHTML = "";

    genres.forEach((genre) => {
        const label = document.createElement("label");
        label.className = "genre-option";

        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = genre.name;

        const text = document.createElement("span");
        text.textContent = `${genre.name} (${genre.count})`;

        label.appendChild(input);
        label.appendChild(text);
        genreFilterList.appendChild(label);
    });
}

function updateFilterSummary() {
    const chips = [];
    if (activeFilters.language) {
        const selectedOption = languageFilter.querySelector(`option[value="${activeFilters.language}"]`);
        chips.push(`<span class="filter-pill">${escapeHtml(selectedOption ? selectedOption.textContent : activeFilters.language)}</span>`);
    }

    activeFilters.genres.forEach((genre) => {
        chips.push(`<span class="filter-pill">${escapeHtml(genre)}</span>`);
    });

    if (chips.length === 0) {
        activeFiltersSummary.style.display = "none";
        activeFiltersSummary.innerHTML = "";
        toggleFiltersBtn.textContent = "Filters";
        return;
    }

    activeFiltersSummary.style.display = "flex";
    activeFiltersSummary.innerHTML = chips.join("");
    toggleFiltersBtn.textContent = `Filters (${chips.length})`;
}

// ===== Search =====
function bindSearchEvents() {
    searchInput.addEventListener("input", () => {
        const query = searchInput.value.trim();
        clearTimeout(searchTimeout);

        if (query.length < 2) {
            searchResults.innerHTML = "";
            searchResults.classList.remove("active");
            return;
        }

        searchTimeout = setTimeout(() => {
            fetchSearchResults(query);
        }, 250);
    });

    document.addEventListener("click", (event) => {
        if (!event.target.closest(".search-box")) {
            searchResults.classList.remove("active");
        }

        if (!event.target.closest(".filters-wrap")) {
            filtersPanel.style.display = "none";
        }
    });

    recommendBtn.addEventListener("click", submitRecommendationRequest);
}

async function fetchSearchResults(query) {
    try {
        const params = new URLSearchParams({ q: query });
        if (activeFilters.language) {
            params.set("language", activeFilters.language);
        }
        if (activeFilters.genres.length > 0) {
            params.set("genres", activeFilters.genres.join(","));
        }

        const res = await fetch(`/api/search?${params.toString()}`);
        const data = await res.json();
        renderSearchResults(data.results || []);
    } catch (err) {
        console.error("Search failed:", err);
    }
}

function renderSearchResults(results) {
    searchResults.innerHTML = "";

    if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result-item"><span class="title">No movies found</span></div>';
        searchResults.classList.add("active");
        return;
    }

    const selectedIndices = new Set(selectedMovies.map((movie) => movie.index));

    results.forEach((movie) => {
        const item = document.createElement("div");
        item.className = `search-result-item ${selectedIndices.has(movie.index) ? "selected" : ""}`;

        const genres = Array.isArray(movie.genres) ? movie.genres.join(", ") : movie.genres;
        const language = movie.language ? movie.language.toUpperCase() : "N/A";

        item.innerHTML = `
            <div class="title">${escapeHtml(movie.title)} (${movie.release_year})${selectedIndices.has(movie.index) ? " ✓" : ""}</div>
            <div class="meta">${escapeHtml(genres || "Unknown")} | Lang: ${escapeHtml(language)} | Rating: ${movie.vote_average} | Popularity: ${movie.popularity}</div>
        `;

        item.addEventListener("click", async () => {
            if (selectedMovies.some((selected) => selected.index === movie.index)) {
                return;
            }

            selectedMovies.push({
                index: movie.index,
                title: movie.title,
                release_year: movie.release_year,
                genres: movie.genres,
            });

            updateSelectedMovies();
            searchInput.value = "";
            searchResults.classList.remove("active");

            if (currentUser) {
                const saved = await saveWatchHistory(movie, null);
                if (saved) {
                    enqueueRatingPrompt({
                        index: movie.index,
                        title: movie.title,
                        release_year: movie.release_year,
                        onRated: (rating) => updateSelectedMovieRating(movie.index, rating),
                    });
                }
            }
        });

        searchResults.appendChild(item);
    });

    searchResults.classList.add("active");
}

function updateSelectedMovies() {
    selectedCountEl.textContent = `(${selectedMovies.length})`;

    if (selectedMovies.length === 0) {
        selectedMoviesEl.innerHTML = '<p class="empty-msg">No movies selected yet. Search and click to add movies.</p>';
        recommendBtn.disabled = true;
        return;
    }

    selectedMoviesEl.innerHTML = selectedMovies.map((movie, index) => `
        <div class="movie-chip">
            ${escapeHtml(movie.title)} (${movie.release_year})
            ${movie.rating ? `<span class="chip-rating">${movie.rating}/${STAR_COUNT}★</span>` : ""}
            <button class="remove-btn" data-index="${index}" title="Remove">&times;</button>
        </div>
    `).join("");

    recommendBtn.disabled = false;

    selectedMoviesEl.querySelectorAll(".remove-btn").forEach((button) => {
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            const movieIndex = Number(button.dataset.index);
            selectedMovies.splice(movieIndex, 1);
            updateSelectedMovies();
        });
    });
}

// ===== Recommendations =====
async function submitRecommendationRequest() {
    if (selectedMovies.length === 0) {
        return;
    }

    loadingEl.style.display = "block";
    recommendBtn.disabled = true;
    recommendationsSection.style.display = "none";

    try {
        const payload = {
            movies: selectedMovies.map((movie) => ({
                index: movie.index,
                title: movie.title,
                release_year: movie.release_year,
                rating: movie.rating,
            })),
            top_k: 10,
            language: activeFilters.language,
            genres: activeFilters.genres,
        };

        const res = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (!res.ok || data.error) {
            alert(data.error || "Failed to get recommendations.");
            return;
        }

        renderWeightInfo(data.weight_info || null);
        renderRecommendations(data.recommendations || []);

        if (currentUser && Array.isArray(data.rating_prompts) && data.rating_prompts.length > 0) {
            enqueueRatingPrompts(data.rating_prompts);
        }
    } catch (err) {
        console.error("Recommendation failed:", err);
        alert("Failed to get recommendations. Make sure the server is running.");
    } finally {
        loadingEl.style.display = "none";
        recommendBtn.disabled = selectedMovies.length === 0;
    }
}

function renderWeightInfo(info) {
    if (!info) {
        weightInfoEl.style.display = "none";
        weightInfoEl.innerHTML = "";
        return;
    }

    weightInfoEl.innerHTML = `
        <div class="regime-label">${escapeHtml(info.regime.replace(/_/g, " "))}</div>
        <div>${escapeHtml(info.regime_description)}</div>
        <div class="weight-context">
            Signals: ${info.total_watched} | Current selection: ${info.current_selection_count || 0} | Recent: ${info.recent_2mo}
        </div>
        <div class="weight-bars">
            ${renderWeightBar("Collaborative", "collab", info.alpha)}
            ${renderWeightBar("Content", "content", info.beta)}
            ${renderWeightBar("Sequential", "sequential", info.gamma)}
            ${renderWeightBar("Popularity", "popularity", info.delta)}
        </div>
    `;
    weightInfoEl.style.display = "block";
}

function renderWeightBar(label, key, value) {
    const percent = Math.round((value || 0) * 100);
    return `
        <div class="weight-bar">
            <span class="weight-bar-label">${label}</span>
            <span class="weight-bar-track">
                <span class="weight-bar-fill ${key}" style="width:${percent}%"></span>
            </span>
            <span class="weight-bar-pct">${percent}%</span>
        </div>
    `;
}

function renderRecommendations(recommendations) {
    lastRecommendations = recommendations;
    recommendationsEl.innerHTML = "";

    if (recommendations.length === 0) {
        recommendationsEl.innerHTML = "<p>No recommendations found with the current filters.</p>";
        recommendationsSection.style.display = "block";
        return;
    }

    recommendations.forEach((recommendation) => {
        const card = document.createElement("div");
        card.className = "rec-card";

        const genres = Array.isArray(recommendation.genres)
            ? recommendation.genres
            : String(recommendation.genres || "").split(",").map((genre) => genre.trim()).filter(Boolean);

        const actionLabel = currentUser ? "Mark as Watched" : "Log in to track";

        card.innerHTML = `
            <div class="rec-rank">${recommendation.rank}</div>
            <div class="rec-info">
                <div class="rec-title">${escapeHtml(recommendation.title)}</div>
                <div class="rec-meta">
                    <span>Year: ${recommendation.release_year}</span>
                    <span>Rating: ${recommendation.vote_average}</span>
                    <span>Popularity: ${recommendation.popularity}</span>
                    <span>Lang: ${escapeHtml((recommendation.language || "N/A").toUpperCase())}</span>
                </div>
                <div class="rec-genres">
                    ${genres.map((genre) => `<span class="genre-tag">${escapeHtml(genre)}</span>`).join("")}
                </div>
                <div class="rec-score">Score: ${recommendation.score}</div>
                <div class="rec-actions">
                    <button class="btn-watched watch-btn" data-index="${recommendation.index}">${actionLabel}</button>
                </div>
            </div>
        `;

        const watchButton = card.querySelector(".watch-btn");
        watchButton.addEventListener("click", async () => {
            if (!currentUser) {
                openAuthModal("login");
                return;
            }

            const movie = {
                index: recommendation.index,
                title: recommendation.title,
                release_year: recommendation.release_year,
            };
            await markMovieAsWatched(movie, watchButton);
        });

        recommendationsEl.appendChild(card);
    });

    recommendationsSection.style.display = "block";
    recommendationsSection.scrollIntoView({ behavior: "smooth" });
}

async function markMovieAsWatched(movie, buttonEl) {
    const saved = await saveWatchHistory(movie, null);
    if (!saved) {
        return;
    }

    updateWatchButtonState(buttonEl, null);
    enqueueRatingPrompt({ ...movie, buttonEl });
}

function updateWatchButtonState(buttonEl, rating) {
    buttonEl.disabled = true;
    buttonEl.classList.add("is-watched");
    buttonEl.textContent = rating
        ? `Watched • ${rating}/${STAR_COUNT}★`
        : "Watched";
}

async function saveWatchHistory(movie, rating) {
    try {
        const res = await fetch("/api/history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                index: movie.index,
                title: movie.title,
                release_year: movie.release_year,
                rating,
            }),
        });

        const data = await res.json();
        if (!res.ok) {
            if (res.status === 401) {
                openAuthModal("login");
            } else {
                alert(data.error || "Could not mark movie as watched.");
            }
            return false;
        }

        return true;
    } catch (err) {
        return false;
    }
}

async function saveMovieRating(movieIndex, rating) {
    try {
        const res = await fetch(`/api/history/${movieIndex}/rating`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rating }),
        });

        const data = await res.json();
        if (!res.ok) {
            if (res.status === 401) {
                openAuthModal("login");
            } else {
                alert(data.error || "Could not save rating.");
            }
            return false;
        }

        return true;
    } catch (err) {
        return false;
    }
}

function updateSelectedMovieRating(movieIndex, rating) {
    const movie = selectedMovies.find((selected) => selected.index === movieIndex);
    if (!movie) {
        return;
    }

    movie.rating = rating;
    updateSelectedMovies();
}

// ===== Rating Queue =====
function bindRatingEvents() {
    ratingModalClose.addEventListener("click", skipCurrentRatingPrompt);
    ratingModal.addEventListener("click", (event) => {
        if (event.target === ratingModal) {
            skipCurrentRatingPrompt();
        }
    });

    Array.from(starRatingEl.querySelectorAll(".star")).forEach((star) => {
        star.addEventListener("mouseenter", () => {
            setRatingStars(Number(star.dataset.value));
        });
        star.addEventListener("click", () => {
            currentRatingValue = Number(star.dataset.value);
            setRatingStars(currentRatingValue);
            ratingValueEl.textContent = `${currentRatingValue} out of ${STAR_COUNT} stars`;
        });
    });

    starRatingEl.addEventListener("mouseleave", () => {
        setRatingStars(currentRatingValue);
    });

    ratingSubmitBtn.addEventListener("click", submitCurrentRating);
    ratingSkipBtn.addEventListener("click", (event) => {
        event.preventDefault();
        skipCurrentRatingPrompt();
    });
}

function enqueueRatingPrompts(movies) {
    movies.forEach((movie) => enqueueRatingPrompt(movie));
}

function enqueueRatingPrompt(movie) {
    const alreadyActive = currentRatingContext && currentRatingContext.index === movie.index;
    const alreadyQueued = ratingQueue.some((queuedMovie) => queuedMovie.index === movie.index);
    if (alreadyActive || alreadyQueued) {
        return;
    }

    ratingQueue.push(movie);
    processRatingQueue();
}

function processRatingQueue() {
    if (currentRatingContext || ratingQueue.length === 0) {
        return;
    }

    currentRatingContext = ratingQueue.shift();
    currentRatingValue = 0;
    ratingMovieTitle.textContent = `${currentRatingContext.title} (${currentRatingContext.release_year})`;
    ratingValueEl.textContent = "Select a rating (1-5 stars)";
    setRatingStars(0);
    ratingModal.style.display = "flex";
}

function setRatingStars(value) {
    Array.from(starRatingEl.querySelectorAll(".star")).forEach((star) => {
        const starValue = Number(star.dataset.value);
        star.classList.toggle("active", starValue <= value);
    });
}

async function submitCurrentRating() {
    if (!currentRatingContext) {
        return;
    }

    if (currentRatingValue < 1) {
        ratingValueEl.textContent = "Please choose between 1 and 5 stars.";
        return;
    }

    const saved = await saveMovieRating(currentRatingContext.index, currentRatingValue);
    if (saved && currentRatingContext.buttonEl) {
        updateWatchButtonState(currentRatingContext.buttonEl, currentRatingValue);
    }
    if (saved && typeof currentRatingContext.onRated === "function") {
        currentRatingContext.onRated(currentRatingValue);
    }

    if (saved && historyModal.style.display === "flex") {
        await openHistoryModal();
    }

    closeCurrentRatingPrompt();
}

function skipCurrentRatingPrompt() {
    if (currentRatingContext && currentRatingContext.buttonEl) {
        updateWatchButtonState(currentRatingContext.buttonEl, null);
    }
    closeCurrentRatingPrompt();
}

function closeCurrentRatingPrompt() {
    ratingModal.style.display = "none";
    currentRatingContext = null;
    currentRatingValue = 0;
    setRatingStars(0);
    processRatingQueue();
}

// ===== Helpers =====
function escapeHtml(value) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(String(value)));
    return div.innerHTML;
}
