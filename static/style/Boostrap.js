document.addEventListener("DOMContentLoaded", function () {
    let navLinks = document.querySelectorAll(".nav-link");
    let currentUrl = window.location.pathname.split("/").pop();

    navLinks.forEach(function (link) {
        link.classList.remove("active"); // Hapus active dari semua link
        if (link.getAttribute("href") === currentUrl) {
            link.classList.add("active"); // Tambahkan active hanya ke yang cocok
        }
    });
});

function toggleText() {
    var moreText = document.getElementById('more');
    var dots = document.getElementById('dots');
    var btnText = document.getElementById('readMoreBtn');

    if (moreText.style.display === 'none') {
      moreText.style.display = 'inline';
      dots.style.display = 'none';
      btnText.innerText = ' Tutup';
    } else {
      moreText.style.display = 'none';
      dots.style.display = 'inline';
      btnText.innerText = ' Baca Selengkapnya';
    }
  }