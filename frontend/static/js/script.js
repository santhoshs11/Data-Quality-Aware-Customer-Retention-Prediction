// RetainIQ — Global JS Utilities
// Shared across all pages

const RetainIQ = {
  apiBase: '',

  async post(endpoint, data={}) {
    const res = await fetch(this.apiBase + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return res.json();
  },

  async get(endpoint) {
    const res = await fetch(this.apiBase + endpoint);
    return res.json();
  },

  toast(msg, type='info') {
    const tc = document.getElementById('toastContainer');
    if (!tc) return;
    const t = document.createElement('div');
    t.className = 'toast-cyber';
    const icons = {info:'ℹ️', success:'✅', error:'🚨', warning:'⚠️'};
    t.innerHTML = `<span>${icons[type]||''}</span> ${msg}`;
    tc.appendChild(t);
    setTimeout(() => t.remove(), 4500);
  },

  formatPct(val) { return (val * 100).toFixed(2) + '%'; },
  formatScore(val) { return parseFloat(val).toFixed(1); }
};

// Animate numbers counting up
function animateCounter(el, target, duration=1200) {
  let start = 0;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { start = target; clearInterval(timer); }
    el.textContent = typeof target === 'float' ? start.toFixed(1) : Math.round(start);
  }, 16);
}

// Dark mode toggle (bonus feature)
function toggleDarkMode() {
  document.body.classList.toggle('light-mode');
  localStorage.setItem('theme', document.body.classList.contains('light-mode') ? 'light' : 'dark');
}

window.addEventListener('DOMContentLoaded', () => {
  // Mark active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === path) link.classList.add('active');
    else link.classList.remove('active');
  });
});
