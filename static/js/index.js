window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    var options = {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: true,
      autoplaySpeed: 5000,
    }

    // On phones, leave carousel content in the normal document flow.
    // This avoids mobile slider layout issues and keeps all figures visible.
    if (window.innerWidth > 768) {
      bulmaCarousel.attach('.carousel', options);
    }

    bulmaSlider.attach();
})
