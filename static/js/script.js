// Modern JavaScript for the image board with dark mode support and animations
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initializeTheme();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize lazy loading
    initLazyLoading();
    
    // Initialize drag and drop for reverse search
    initDragAndDrop();
    
    // Add scroll effect to navbar
    initNavbarScroll();

    // Handle smooth navigation
    initSmoothNavigation();
});

// Theme functionality
function initializeTheme() {
    const darkMode = document.body.getAttribute('data-dark-mode') === '1';
    const themeToggle = document.getElementById('themeToggle');
    
    if (themeToggle) {
        // Set initial icon
        updateThemeIcon(darkMode);
        
        // Add click event
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Apply theme to body
    document.body.setAttribute('data-theme', darkMode ? 'dark' : 'light');
}

function initSmoothNavigation() {
    // Check if this is a back navigation
    const isBackNavigation = performance.getEntriesByType('navigation')[0]?.type === 'back_forward';

if (isBackNavigation) {
        document.body.classList.add('no-page-animations');
        
        // Re-enable animations after a short delay for future interactions
        setTimeout(() => {
            document.body.classList.remove('no-page-animations');
        }, 100);
    }
    
    // Add smooth transition for all internal links
    document.addEventListener('click', function(e) {
        const link = e.target.closest('a');
        if (link && link.href && link.href.startsWith(window.location.origin) && !link.href.includes('#') && link.target !== '_blank') {
            e.preventDefault();
            
            // Add fade-out effect
            document.body.classList.add('page-transition-out');
            
            // Navigate after transition
            setTimeout(() => {
                window.location.href = link.href;
            }, 300);
        }
    });
    
    // Handle browser back/forward buttons
    window.addEventListener('popstate', function() {
        document.body.classList.add('page-transition-in');
        setTimeout(() => {
            document.body.classList.remove('page-transition-in');
        }, 300);
    });
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    const newDarkMode = newTheme === 'dark' ? 1 : 0;
    
    // Add transition class for smooth theme change
    document.body.classList.add('theme-transition');
    
    // Update UI immediately
    document.body.setAttribute('data-theme', newTheme);
    updateThemeIcon(newDarkMode === 1);
    
    // Save preference to server
    fetch('/toggle-dark-mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            // Revert if failed
            document.body.setAttribute('data-theme', currentTheme);
            updateThemeIcon(currentTheme === 'dark');
        }
        
        // Remove transition class after animation
        setTimeout(() => {
            document.body.classList.remove('theme-transition');
        }, 500);
    })
    .catch(error => {
        console.error('Error toggling theme:', error);
        document.body.setAttribute('data-theme', currentTheme);
        updateThemeIcon(currentTheme === 'dark');
        document.body.classList.remove('theme-transition');
    });
}

function updateThemeIcon(isDark) {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.innerHTML = isDark ? 
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>' :
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    }
}

// Navbar scroll effect
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    
    let lastScrollY = window.scrollY;
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
            
            // Hide navbar on scroll down, show on scroll up
            if (window.scrollY > lastScrollY && window.scrollY > 100) {
                navbar.style.transform = 'translateY(-100%)';
            } else {
                navbar.style.transform = 'translateY(0)';
            }
        } else {
            navbar.classList.remove('scrolled');
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScrollY = window.scrollY;
    });
}

// Event listeners
function setupEventListeners() {
    // Search input debouncing
    const searchInput = document.querySelector('input[name="q"]');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(function() {
            if (this.value.trim().length > 1) {
                fetchTags(this.value.trim());
            }
        }, 300));
    }
    
    // Image modal functionality
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('image-item') || 
            e.target.closest('.image-item')) {
            const imageId = e.target.closest('.image-item').dataset.imageId;
            if (imageId) {
                openImageModal(imageId);
            }
        }
    });
    
    // Add hover effects to all cards
    const cards = document.querySelectorAll('.image-item, .result-item, .glass-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.classList.add('hover-lift');
        });
        
        card.addEventListener('mouseleave', () => {
            card.classList.remove('hover-lift');
        });
    });
}

// Lazy loading
function initLazyLoading() {
    if ('IntersectionObserver' in window) {
        const lazyImages = document.querySelectorAll('img[loading="lazy"]');
        
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.removeAttribute('data-src');
                    
                    // Add fade-in effect
                    img.style.opacity = '0';
                    img.style.transition = 'opacity 0.5s ease';
                    
                    setTimeout(() => {
                        img.style.opacity = '1';
                    }, 50);
                    
                    imageObserver.unobserve(img);
                }
            });
        }, {
            rootMargin: '0px 0px 200px 0px' // Load images 200px before they enter viewport
        });
        
        lazyImages.forEach(img => {
            if (img.hasAttribute('data-src')) {
                imageObserver.observe(img);
            }
        });
    }
}

// Drag and drop for reverse search
function initDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    if (dropZone && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        // Handle file input change
        fileInput.addEventListener('change', handleFileSelect, false);
        
        // Click on drop zone to trigger file input
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    document.getElementById('dropZone').classList.add('highlight');
}

function unhighlight() {
    document.getElementById('dropZone').classList.remove('highlight');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.match('image.*')) {
            uploadFileForSearch(file);
        } else {
            showFlash('Please select an image file.', 'error');
        }
    }
}

function uploadFileForSearch(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const topKSelect = document.getElementById('top_k');
    if (topKSelect) {
        formData.append('top_k', topKSelect.value);
    }
    
    const resultsContainer = document.getElementById('resultsContainer');
    if (resultsContainer) {
        resultsContainer.innerHTML = '<div class="loading">Searching for similar images...</div>';
    }
    
    fetch('{{ url_for("reverse_search") }}', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(html => {
        if (resultsContainer) {
            resultsContainer.innerHTML = html;
            
            // Add animation to results
            const resultItems = resultsContainer.querySelectorAll('.result-item');
            resultItems.forEach((item, index) => {
                item.style.animationDelay = `${index * 0.1}s`;
                item.classList.add('fadeIn');
            });
        }
    })
    .catch(error => {
        if (resultsContainer) {
            resultsContainer.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
        console.error('Error:', error);
    });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

function fetchTags(query) {
    fetch(`/api/tags?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(tags => {
            // Implement tag suggestions UI here
            console.log('Available tags:', tags);
        })
        .catch(error => {
            console.error('Error fetching tags:', error);
        });
}

function showFlash(message, type) {
    // Create flash message element
    const flash = document.createElement('div');
    flash.className = `flash ${type}`;
    flash.textContent = message;
    
    // Add to page
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(flash, container.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            flash.style.opacity = '0';
            flash.style.transition = 'opacity 0.5s ease';
            
            setTimeout(() => {
                flash.remove();
            }, 500);
        }, 5000);
    }
}

// Image modal (for future enhancement)
function openImageModal(imageId) {
    // This would open a modal with the full-size image
    // Implementation depends on your modal library of choice
    console.log('Open image modal for ID:', imageId);
}

// Add CSS for theme transition
const style = document.createElement('style');
style.textContent = `
    .theme-transition {
        transition: background-color 0.5s ease, color 0.5s ease !important;
    }
    
    .theme-transition * {
        transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease !important;
    }
`;
document.head.appendChild(style);

// Enhanced image loading with error handling
function initImageLoading() {
    const images = document.querySelectorAll('.image-item img');
    
    images.forEach(img => {
        // Skip if already loaded
        if (img.complete && img.naturalHeight !== 0) {
            img.style.opacity = '1';
            const loadingElement = img.previousElementSibling;
            if (loadingElement && loadingElement.classList.contains('image-loading')) {
                loadingElement.style.display = 'none';
            }
            return;
        }
        
        // Handle image errors
        img.addEventListener('error', function() {
            this.style.opacity = '1';
            const loadingElement = this.previousElementSibling;
            if (loadingElement && loadingElement.classList.contains('image-loading')) {
                loadingElement.style.display = 'none';
            }
            
            // Show placeholder for broken images
            this.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjFmNWY5Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5NGEzYjgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+';
        });
    });
}

// Call this function in your DOMContentLoaded event
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    initImageLoading();
});