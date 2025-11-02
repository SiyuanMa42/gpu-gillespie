// GPU-Gillespie Website Main JavaScript
// Modern scientific computing website with animations and interactions

class GPUWebsite {
    constructor() {
        this.init();
        this.setupScrollAnimations();
        this.setupParticleSystem();
        this.setupPerformanceChart();
        this.setupTypingAnimation();
        this.setupScrollIndicator();
    }

    init() {
        // Initialize Splitting.js for text animations
        if (typeof Splitting !== 'undefined') {
            Splitting();
        }

        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Add intersection observer for animations
        this.setupIntersectionObserver();
    }

    setupIntersectionObserver() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        // Observe all feature cards and model cards
        document.querySelectorAll('.feature-card, .model-card').forEach(el => {
            observer.observe(el);
        });
    }

    setupScrollAnimations() {
        // Parallax effect for hero background
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const hero = document.querySelector('.hero-bg');
            
            if (hero) {
                const speed = scrolled * 0.5;
                hero.style.transform = `translateY(${speed}px)`;
            }

            // Animate metric cards on scroll
            const metricCards = document.querySelectorAll('.metric-card');
            metricCards.forEach((card, index) => {
                const rect = card.getBoundingClientRect();
                const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
                
                if (isVisible) {
                    card.style.transform = 'translateY(0)';
                    card.style.opacity = '1';
                }
            });
        });
    }

    setupParticleSystem() {
        // P5.js particle system for hero background
        if (typeof p5 !== 'undefined') {
            new p5((p) => {
                let particles = [];
                let numParticles = 50;

                p.setup = () => {
                    const canvas = p.createCanvas(window.innerWidth, window.innerHeight);
                    canvas.parent('particleCanvas');
                    canvas.style('position', 'absolute');
                    canvas.style('top', '0');
                    canvas.style('left', '0');
                    canvas.style('z-index', '1');

                    // Initialize particles
                    for (let i = 0; i < numParticles; i++) {
                        particles.push({
                            x: p.random(p.width),
                            y: p.random(p.height),
                            vx: p.random(-1, 1),
                            vy: p.random(-1, 1),
                            size: p.random(2, 6),
                            opacity: p.random(0.3, 0.8)
                        });
                    }
                };

                p.draw = () => {
                    p.clear();
                    
                    // Update and draw particles
                    particles.forEach(particle => {
                        // Update position
                        particle.x += particle.vx;
                        particle.y += particle.vy;

                        // Wrap around edges
                        if (particle.x < 0) particle.x = p.width;
                        if (particle.x > p.width) particle.x = 0;
                        if (particle.y < 0) particle.y = p.height;
                        if (particle.y > p.height) particle.y = 0;

                        // Draw particle
                        p.fill(45, 125, 142, particle.opacity * 255);
                        p.noStroke();
                        p.circle(particle.x, particle.y, particle.size);

                        // Draw connections to nearby particles
                        particles.forEach(other => {
                            const distance = p.dist(particle.x, particle.y, other.x, other.y);
                            if (distance < 100) {
                                const alpha = p.map(distance, 0, 100, 0.3, 0);
                                p.stroke(45, 125, 142, alpha * 255);
                                p.strokeWeight(1);
                                p.line(particle.x, particle.y, other.x, other.y);
                            }
                        });
                    });
                };

                p.windowResized = () => {
                    p.resizeCanvas(window.innerWidth, window.innerHeight);
                };
            });
        }
    }

    setupPerformanceChart() {
        // ECharts performance visualization
        if (typeof echarts !== 'undefined') {
            const chartElement = document.getElementById('performanceChart');
            if (chartElement) {
                const chart = echarts.init(chartElement);
                
                const option = {
                    backgroundColor: 'transparent',
                    grid: {
                        left: '10%',
                        right: '10%',
                        top: '10%',
                        bottom: '15%'
                    },
                    xAxis: {
                        type: 'category',
                        data: ['100', '1K', '5K', '10K', '50K', '100K'],
                        axisLine: {
                            lineStyle: { color: '#718096' }
                        },
                        axisLabel: {
                            color: '#718096',
                            fontSize: 12
                        }
                    },
                    yAxis: {
                        type: 'value',
                        name: 'Speedup Factor',
                        nameTextStyle: { color: '#718096' },
                        axisLine: {
                            lineStyle: { color: '#718096' }
                        },
                        axisLabel: {
                            color: '#718096',
                            fontSize: 12
                        },
                        splitLine: {
                            lineStyle: { color: '#2d7d8e', opacity: 0.2 }
                        }
                    },
                    series: [{
                        name: 'GPU Speedup',
                        type: 'line',
                        data: [8.5, 50.8, 75.2, 107.6, 112.3, 116.8],
                        lineStyle: {
                            color: '#2d7d8e',
                            width: 3
                        },
                        itemStyle: {
                            color: '#2d7d8e',
                            borderWidth: 2,
                            borderColor: '#ffffff'
                        },
                        areaStyle: {
                            color: {
                                type: 'linear',
                                x: 0, y: 0, x2: 0, y2: 1,
                                colorStops: [
                                    { offset: 0, color: 'rgba(45, 125, 142, 0.3)' },
                                    { offset: 1, color: 'rgba(45, 125, 142, 0.05)' }
                                ]
                            }
                        },
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 8
                    }],
                    tooltip: {
                        trigger: 'axis',
                        backgroundColor: 'rgba(26, 54, 93, 0.9)',
                        borderColor: '#2d7d8e',
                        textStyle: { color: '#ffffff' },
                        formatter: function(params) {
                            return `轨迹数: ${params[0].name}<br/>加速比: ${params[0].value}x`;
                        }
                    }
                };

                chart.setOption(option);

                // Animate chart on scroll
                const observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            chart.resize();
                        }
                    });
                });
                observer.observe(chartElement);

                // Resize chart on window resize
                window.addEventListener('resize', () => {
                    chart.resize();
                });
            }
        }
    }

    setupTypingAnimation() {
        // Enhanced typing animation with random pause
        const typingElement = document.querySelector('.typing-animation');
        if (typingElement) {
            const texts = [
                '高性能随机模拟与GPU加速',
                '计算生物学的并行计算解决方案',
                '高达100倍的性能提升',
                '支持数千个轨迹的并行计算'
            ];
            
            let textIndex = 0;
            let charIndex = 0;
            let isDeleting = false;
            
            const typeText = () => {
                const currentText = texts[textIndex];
                
                if (isDeleting) {
                    typingElement.textContent = currentText.substring(0, charIndex - 1);
                    charIndex--;
                } else {
                    typingElement.textContent = currentText.substring(0, charIndex + 1);
                    charIndex++;
                }

                let typeSpeed = isDeleting ? 50 : 100;
                
                // Add random pause at end of words
                if (!isDeleting && charIndex === currentText.length) {
                    typeSpeed = 2000;
                    isDeleting = true;
                } else if (isDeleting && charIndex === 0) {
                    isDeleting = false;
                    textIndex = (textIndex + 1) % texts.length;
                    typeSpeed = 500;
                }

                setTimeout(typeText, typeSpeed);
            };

            // Start typing animation after a delay
            setTimeout(typeText, 1000);
        }
    }

    setupScrollIndicator() {
        const indicator = document.getElementById('scrollIndicator');
        if (indicator) {
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                const maxHeight = document.documentElement.scrollHeight - window.innerHeight;
                const scrollPercent = scrolled / maxHeight;
                
                indicator.style.transform = `scaleX(${scrollPercent})`;
            });
        }
    }

    // Utility methods for animations
    animateValue(element, start, end, duration) {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (end - start) * this.easeOutCubic(progress);
            element.textContent = Math.round(current);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // Performance monitoring
    measurePerformance() {
        if (window.performance && window.performance.timing) {
            const timing = window.performance.timing;
            const loadTime = timing.loadEventEnd - timing.navigationStart;
            console.log(`Page load time: ${loadTime}ms`);
        }
    }
}

// Initialize website when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const website = new GPUWebsite();
    
    // Add some interactive features
    
    // Feature cards hover effect
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            if (typeof anime !== 'undefined') {
                anime({
                    targets: card,
                    scale: 1.02,
                    duration: 300,
                    easing: 'easeOutCubic'
                });
            }
        });
        
        card.addEventListener('mouseleave', () => {
            if (typeof anime !== 'undefined') {
                anime({
                    targets: card,
                    scale: 1,
                    duration: 300,
                    easing: 'easeOutCubic'
                });
            }
        });
    });

    // Model cards click effect
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', () => {
            // Add a subtle click animation
            if (typeof anime !== 'undefined') {
                anime({
                    targets: card,
                    scale: [1, 0.98, 1],
                    duration: 200,
                    easing: 'easeInOutQuad'
                });
            }
        });
    });

    // Add loading animation for images
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('load', () => {
            if (typeof anime !== 'undefined') {
                anime({
                    targets: img,
                    opacity: [0, 1],
                    duration: 500,
                    easing: 'easeOutCubic'
                });
            }
        });
    });

    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'PageDown') {
            e.preventDefault();
            window.scrollBy(0, window.innerHeight);
        } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
            e.preventDefault();
            window.scrollBy(0, -window.innerHeight);
        }
    });

    // Measure performance
    website.measurePerformance();
});

// Export for potential external use
window.GPUWebsite = GPUWebsite;