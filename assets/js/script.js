'use strict';



// element toggle function
const elementToggleFunc = function (elem) { elem.classList.toggle("active"); }



// sidebar variables
const sidebar = document.querySelector("[data-sidebar]");
const sidebarBtn = document.querySelector("[data-sidebar-btn]");

// sidebar toggle functionality for mobile
sidebarBtn.addEventListener("click", function () {
  elementToggleFunc(sidebar);
  this.querySelector("span").innerText = sidebar.classList.contains("active") ? "Hide Contacts" : "Show Contacts";
});



// custom select variables (mobile) and filter buttons (larger screens)
const select = document.querySelector("[data-select]");
const selectItems = document.querySelectorAll("[data-select-item]");
const selectValue = document.querySelector("[data-select-value]");
const filterBtn = document.querySelectorAll("[data-filter-btn]");
const filterItems = document.querySelectorAll("[data-filter-item]");

select.addEventListener("click", function () { elementToggleFunc(this); });

const filterFunc = function (selectedValue) {
  for (const item of filterItems) {
    item.classList.toggle("active", selectedValue === "all" || selectedValue === item.dataset.category);
  }
}

// keeps the mobile select and the desktop buttons showing the same category
const applyFilter = function (label) {
  selectValue.innerText = label;
  filterFunc(label.toLowerCase());
  for (const btn of filterBtn) btn.classList.toggle("active", btn.innerText === label);
}

for (const item of selectItems) {
  item.addEventListener("click", function () {
    applyFilter(this.innerText);
    elementToggleFunc(select);
  });
}

for (const btn of filterBtn) {
  btn.addEventListener("click", function () { applyFilter(this.innerText); });
}



// page navigation variables
const navigationLinks = document.querySelectorAll("[data-nav-link]");
const pages = document.querySelectorAll("[data-page]");

for (const link of navigationLinks) {
  link.addEventListener("click", function () {
    const page = this.innerText.toLowerCase();
    for (const p of pages) p.classList.toggle("active", p.dataset.page === page);
    for (const l of navigationLinks) l.classList.toggle("active", l === this);
    window.scrollTo(0, 0);
  });
}
