// Progressive enhancement only: the page is complete without JavaScript.

// pointer spotlight (fine pointers only, so touch screens skip it)
if (matchMedia('(pointer: fine)').matches && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const root = document.documentElement;
  addEventListener('pointermove', e => {
    root.style.setProperty('--x', e.clientX + 'px');
    root.style.setProperty('--y', e.clientY + 'px');
  }, { passive: true });
}

// highlight the nav link of the section currently in view
const links = new Map([...document.querySelectorAll('.nav a')].map(a => [a.hash.slice(1), a]));
if (links.size) {
  const observer = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      for (const [id, a] of links) a.setAttribute('aria-current', id === e.target.id);
    }
  }, { rootMargin: '-35% 0px -60% 0px' });
  for (const id of links.keys()) {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  }
}
