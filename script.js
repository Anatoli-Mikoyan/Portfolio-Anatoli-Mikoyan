/* ==========================================================================
   Portfolio — Anatoli Mikoyan
   Interactions : thème, navigation, révélations, compteurs, formulaire.
   ========================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------- Thème */
  const root = document.documentElement;
  const stored = localStorage.getItem("theme");
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  root.setAttribute("data-theme", stored || (prefersLight ? "light" : "dark"));

  $("#themeToggle")?.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    $('meta[name="theme-color"]')?.setAttribute("content", next === "dark" ? "#0a0f1c" : "#f4f6fb");
  });

  /* -------------------------------------------------------- Menu mobile */
  const burger = $("#burger");
  const navLinks = $("#navLinks");

  burger?.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    burger.setAttribute("aria-expanded", String(open));
    burger.innerHTML = open ? '<i class="bx bx-x"></i>' : '<i class="bx bx-menu"></i>';
  });

  const closeMenu = () => {
    navLinks?.classList.remove("open");
    if (burger) {
      burger.setAttribute("aria-expanded", "false");
      burger.innerHTML = '<i class="bx bx-menu"></i>';
    }
  };
  $$("#navLinks a").forEach((a) => a.addEventListener("click", closeMenu));

  /* ------------------------------------ Barre de progression + nav collée */
  const nav = $("#nav");
  const progress = $("#progress");
  const toTop = $("#toTop");

  const onScroll = () => {
    const y = window.scrollY;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav?.classList.toggle("scrolled", y > 24);
    toTop?.classList.toggle("show", y > 600);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------------------------------------------- Lien de section actif */
  const sections = $$("main section[id]");
  if (sections.length) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          $$("#navLinks a").forEach((a) => a.classList.remove("active"));
          $(`#navLinks a[href="#${e.target.id}"]`)?.classList.add("active");
        });
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: 0 }
    );
    sections.forEach((s) => spy.observe(s));
  }

  /* --------------------------------- Déclenchement à l'entrée dans l'écran

     IntersectionObserver seul ne suffit pas : ses callbacks sont regroupés par
     frame, donc un élément qui traverse entièrement l'écran entre deux frames
     (molette rapide, inertie sur mobile, saut vers une ancre) n'est jamais
     signalé — et reste figé dans son état initial.

     On teste donc soi-même la position, à chaque frame de défilement. La
     condition « le haut de l'élément est passé sous la ligne de déclenchement »
     est vraie aussi bien à l'entrée par le bas qu'après l'avoir dépassé, donc
     rien ne peut être manqué, quelle que soit la vitesse. */
  const whenInView = (elements, activate, offset = 60) => {
    const pending = new Set(elements);
    if (!pending.size) return;

    const flush = () => {
      const line = window.innerHeight - offset;
      for (const el of pending) {
        if (el.getBoundingClientRect().top < line) {
          activate(el);
          pending.delete(el);
        }
      }
      if (!pending.size) {
        window.removeEventListener("scroll", schedule);
        window.removeEventListener("resize", schedule);
      }
    };

    let queued = false;
    const schedule = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        flush();
      });
    };

    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    schedule();
  };

  /* ------------------------------------------------ Révélation au scroll */
  whenInView($$(".reveal"), (el) => el.classList.add("visible"), 40);

  /* --------------------------------------------------- Compteurs animés
     La valeur finale est déjà dans le HTML : on ne repart de 0 que le temps
     de l'animation, jamais avant. */
  whenInView($$("[data-count]"), (el) => {
    if (reduceMotion) return;
    const target = Number(el.dataset.count) || 0;
    const suffix = el.dataset.suffix || "";

    const duration = 1100;
    const start = performance.now();
    const tick = (now) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  /* ---------------------------------------- Halo lumineux sur les cartes */
  if (window.matchMedia("(hover: hover)").matches) {
    $$(".card").forEach((card) => {
      card.addEventListener("pointermove", (e) => {
        const r = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${e.clientX - r.left}px`);
        card.style.setProperty("--my", `${e.clientY - r.top}px`);
      });
    });
  }

  /* ------------------------------------------ Bandeau d'outils en boucle */
  const marquee = $("#marquee");
  if (marquee) marquee.innerHTML += marquee.innerHTML;

  /* ------------------------------------------------ Intitulés animés (hero) */
  const roles = [
    "technicien support N1 / N2",
    "technicien de proximité",
    "technicien poste de travail",
    "technicien déploiement",
  ];
  const typedTarget = $("#typed");

  if (typedTarget && !reduceMotion) {
    const caret = document.createElement("span");
    caret.className = "typed-cursor";
    caret.textContent = "|";
    typedTarget.after(caret);

    let i = 0;
    let pos = 0;
    let deleting = false;

    const tick = () => {
      const chars = [...roles[i]];
      pos += deleting ? -1 : 1;
      typedTarget.textContent = chars.slice(0, pos).join("");

      let delay = deleting ? 30 : 60;
      if (!deleting && pos === chars.length) {
        delay = 2000;
        deleting = true;
      } else if (deleting && pos === 0) {
        deleting = false;
        i = (i + 1) % roles.length;
        delay = 320;
      }
      setTimeout(tick, delay);
    };
    setTimeout(tick, 500);
  } else if (typedTarget) {
    typedTarget.textContent = roles[0];
  }

  /* ------------------------------------------------ Formulaire → mailto */
  $("#contactForm")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    const name = f.name.value.trim();
    const company = f.company.value.trim();
    const subject = f.subject.value.trim();
    const message = f.message.value.trim();

    const body = [message, "", "—", name + (company ? ` — ${company}` : "")].join("\n");

    window.location.href =
      "mailto:anatoli.mikoyan95@gmail.com" +
      `?subject=${encodeURIComponent(subject)}` +
      `&body=${encodeURIComponent(body)}`;
  });

  /* ----------------------------------------------------- Année du footer */
  const year = $("#year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
