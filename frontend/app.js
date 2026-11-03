// Moods shown on the dashboard
const MOODS = [
  { id: 'sunny', label: 'Sunny', hint: 'bright and open' },
  { id: 'tender', label: 'Tender', hint: 'soft and close' },
  { id: 'focused', label: 'Focused', hint: 'deep and steady' },
  { id: 'restless', label: 'Restless', hint: 'moving and electric' },
  { id: 'lowkey', label: 'Low-key', hint: 'quiet and hazy' },
  { id: 'nostalgic', label: 'Nostalgic', hint: 'warm and familiar' }
];

const EVENT_LABELS = { like: 'Liked', skip: 'Skipped' };

// Elements
const loginView = document.querySelector('#login-view');
const dashboardView = document.querySelector('#dashboard-view');
const loginForm = document.querySelector('#login-form');
const emailInput = document.querySelector('#email');
const passwordInput = document.querySelector('#password');
const formError = document.querySelector('#form-error');
const discoverPanel = document.querySelector('#discover-panel');
const historyPanel = document.querySelector('#history-panel');
const tabDiscover = document.querySelector('#tab-discover');
const tabHistory = document.querySelector('#tab-history');
const moodGrid = document.querySelector('#mood-grid');
const resultsTitle = document.querySelector('#results-title');
const statusText = document.querySelector('#status');
const trackList = document.querySelector('#track-list');
const moreButton = document.querySelector('#more-button');
const historyStatus = document.querySelector('#history-status');
const historyList = document.querySelector('#history-list');

let activeMood = null;

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// Talks to the Flask backend. Returns { ok, status, data }. Throws if the server can't be reached.
async function api(path, { method = 'GET', body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    // Response had no JSON body
  }
  return { ok: response.ok, status: response.status, data };
}

// ---------- Song rows (shared by recommendations and history)

// Dataset songs only have a Spotify track id, so show Spotify's own player
function makePlayer(trackId) {
  const frame = document.createElement('iframe');
  frame.src = `https://open.spotify.com/embed/track/${encodeURIComponent(trackId)}`;
  frame.title = 'Spotify player';
  frame.height = '80';
  frame.loading = 'lazy';
  frame.allow = 'encrypted-media';
  return frame;
}

async function sendFeedback(trackId, action) {
  try {
    const result = await api('/api/feedback', { method: 'POST', body: { track_id: trackId, action } });
    if (result.status === 401) {
      showLogin('Your session ended. Please log in again.');
      return false;
    }
    return result.ok;
  } catch {
    statusText.textContent = 'Could not reach the server.';
    return false;
  }
}

function makeActions(trackId, row) {
  const actions = makeEl('div', 'track-actions');
  const like = makeEl('button', 'action-button', 'Like');
  like.type = 'button';
  like.setAttribute('aria-pressed', 'false');
  const skip = makeEl('button', 'action-button', 'Skip');
  skip.type = 'button';

  like.addEventListener('click', async () => {
    const wasLiked = like.getAttribute('aria-pressed') === 'true';
    if (await sendFeedback(trackId, wasLiked ? 'clear' : 'like')) {
      like.setAttribute('aria-pressed', String(!wasLiked));
      like.textContent = wasLiked ? 'Like' : 'Liked';
    }
  });

  skip.addEventListener('click', async () => {
    if (await sendFeedback(trackId, 'skip')) row.remove();
  });

  actions.append(like, skip);
  return actions;
}

// ---------- Moods and recommendations

function renderMoods() {
  MOODS.forEach((mood) => {
    const button = makeEl('button', 'mood-button');
    button.type = 'button';
    button.dataset.mood = mood.id;
    button.setAttribute('aria-pressed', 'false');
    button.append(makeEl('strong', '', mood.label), makeEl('span', '', mood.hint));
    button.addEventListener('click', () => selectMood(mood.id));
    moodGrid.append(button);
  });
}

function renderTrack(song) {
  const row = makeEl('li', 'track-row');
  row.append(makePlayer(song.id), makeActions(song.id, row));
  trackList.append(row);
}

async function selectMood(moodId) {
  activeMood = moodId;
  const mood = MOODS.find((m) => m.id === moodId);

  moodGrid.querySelectorAll('.mood-button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.mood === moodId));
  });

  resultsTitle.textContent = `Songs for a ${mood.label.toLowerCase()} day`;
  statusText.textContent = 'Loading songs...';
  trackList.replaceChildren();
  moreButton.classList.add('is-hidden');

  let result;
  try {
    result = await api(`/api/songs?mood=${encodeURIComponent(moodId)}`);
  } catch {
    statusText.textContent = 'Could not reach the server. Is it running?';
    return;
  }
  if (activeMood !== moodId) return; // user picked another mood while loading

  if (!result.ok) {
    statusText.textContent = result.data?.error || 'Could not load songs.';
    return;
  }

  const songs = result.data;
  if (songs.length === 0) {
    statusText.textContent = 'No new songs found for this mood.';
    return;
  }

  statusText.textContent = songs[0].personalized
    ? 'Picked to match your taste.'
    : 'Like at least 3 songs and your picks will start to match your taste.';
  songs.forEach(renderTrack);
  moreButton.classList.remove('is-hidden');
}

// ---------- History

async function loadHistory() {
  historyStatus.textContent = 'Loading...';
  historyList.replaceChildren();

  let result;
  try {
    result = await api('/api/history');
  } catch {
    historyStatus.textContent = 'Could not reach the server. Is it running?';
    return;
  }
  if (result.status === 401) {
    showLogin('Your session ended. Please log in again.');
    return;
  }
  if (!result.ok) {
    historyStatus.textContent = result.data?.error || 'Could not load your history.';
    return;
  }

  if (result.data.length === 0) {
    historyStatus.textContent = 'Nothing here yet. Like, skip or look up a song and it will show up here.';
    return;
  }

  historyStatus.textContent = '';
  result.data.forEach((item) => {
    const when = new Date(item.at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    const row = makeEl('li', 'track-row');
    row.append(makeEl('p', 'track-note', `${EVENT_LABELS[item.event] || item.event} · ${when}`), makePlayer(item.id));
    historyList.append(row);
  });
}

function showPanel(name) {
  const showHistory = name === 'history';
  discoverPanel.classList.toggle('is-hidden', showHistory);
  historyPanel.classList.toggle('is-hidden', !showHistory);
  tabDiscover.setAttribute('aria-pressed', String(!showHistory));
  tabHistory.setAttribute('aria-pressed', String(showHistory));
  if (showHistory) loadHistory();
}

tabDiscover.addEventListener('click', () => showPanel('discover'));
tabHistory.addEventListener('click', () => showPanel('history'));

// ---------- Login and accounts

function enterApp(email) {
  document.querySelector('#profile-email').textContent = email;
  loginView.classList.add('is-hidden');
  dashboardView.classList.remove('is-hidden');
}

function showLogin(message = '') {
  activeMood = null;
  dashboardView.classList.add('is-hidden');
  loginView.classList.remove('is-hidden');
  loginForm.reset();
  formError.textContent = message;

  showPanel('discover');
  resultsTitle.textContent = '';
  statusText.textContent = '';
  trackList.replaceChildren();
  historyList.replaceChildren();
  moreButton.classList.add('is-hidden');
  moodGrid.querySelectorAll('.mood-button').forEach((button) => button.setAttribute('aria-pressed', 'false'));
}

async function submitAuth(mode) {
  formError.textContent = '';
  const email = emailInput.value.trim();
  const password = passwordInput.value;
  if (!email || password.length < 8) {
    formError.textContent = 'Enter your email and a password of at least 8 characters.';
    return;
  }
  try {
    const result = await api(`/api/${mode}`, { method: 'POST', body: { email, password } });
    if (!result.ok) {
      formError.textContent = result.data?.error || 'Something went wrong. Try again.';
      return;
    }
    enterApp(result.data.email);
  } catch {
    formError.textContent = 'Could not reach the server. Is it running?';
  }
}

loginForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submitAuth('login');
});

document.querySelector('#register-button').addEventListener('click', () => submitAuth('register'));

document.querySelector('#logout-button').addEventListener('click', async () => {
  try {
    await api('/api/logout', { method: 'POST' });
  } catch {
    // Even if the server can't be reached, return to the login screen
  }
  showLogin();
});

moreButton.addEventListener('click', () => {
  if (activeMood) selectMood(activeMood);
});

// ---------- Setup

renderMoods();
document.querySelector('#current-date').textContent = new Intl.DateTimeFormat('en-US', {
  weekday: 'long', month: 'long', day: 'numeric'
}).format(new Date());

// If the user is still logged in from last time, skip the login screen
api('/api/me')
  .then((result) => {
    if (result.ok) enterApp(result.data.email);
  })
  .catch(() => {
    // Server not reachable: stay on the login screen
  });