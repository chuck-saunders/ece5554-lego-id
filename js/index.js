document.addEventListener("DOMContentLoaded", function() {
    const images = document.querySelectorAll(".image");
    let currentIndex = 0;
    let scrollingEnabled = false;  // Flag to allow scrolling past the images
    const totalImages = images.length;
  
    // Initially show the first image
    images[currentIndex].classList.add("active");
  
    // Listen for scroll events
    window.addEventListener("wheel", function(event) {
      if (event.deltaY < 0 && window.scrollY === 0) {
        scrollingEnabled = false;
        images[currentIndex].classList.remove("active");
        currentIndex = 0;
        images[currentIndex].classList.add("active");
      }
      // If scrolling is disabled, flip images and prevent default scroll behavior
      else if (!scrollingEnabled) {
        event.preventDefault();  // Stop default scrolling
        const delta = event.deltaY;
        console.log(window.scrollY, delta)
        if (delta > 0) {
          // Scroll down: show next image
          showNextImage();
        } 
        else {
          // Scroll up: show previous image
          showPreviousImage();
        }
      }
    }, { passive: false }); // passive: false is required to use preventDefault()
  
    function showNextImage() {
      if (currentIndex < totalImages - 1) {
        images[currentIndex].classList.remove("active");
        currentIndex += 1;
        //console.log(scrollingEnabled, currentIndex, event.deltaY);
        images[currentIndex].classList.add("active");
      } else {
        // Enable scrolling after the last image
        scrollingEnabled = true;
      }
    }
  
    function showPreviousImage() {
      if (currentIndex > 0) {
        images[currentIndex].classList.remove("active");
        currentIndex -= 1;
        images[currentIndex].classList.add("active");
      } 
    }
});