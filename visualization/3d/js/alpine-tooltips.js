/**
 * keywords: [alpine, tooltips, tippy, interactive, visualization]
 * 
 * Alpine.js + Tippy.js integration for rich tooltips
 */

document.addEventListener('alpine:init', () => {
    // Configure Tippy defaults for dark theme
    tippy.setDefaultProps({
        theme: 'dark',
        animation: 'fade',
        duration: [200, 150],
        arrow: true,
        inertia: true,
    });
    
    // Magic: $tooltip for programmatic tooltips
    Alpine.magic('tooltip', el => (content, options = {}) => {
        const instance = tippy(el, { 
            content,
            trigger: 'manual',
            ...options 
        });
        
        instance.show();
        
        if (options.duration !== false) {
            setTimeout(() => {
                instance.hide();
                setTimeout(() => instance.destroy(), 150);
            }, options.duration || 2000);
        }
        
        return instance;
    });
    
    // Directive: x-tooltip for declarative tooltips
    Alpine.directive('tooltip', (el, { expression, modifiers }, { evaluate, effect, cleanup }) => {
        let instance = null;
        
        const options = {
            placement: modifiers.includes('top') ? 'top' : 
                       modifiers.includes('bottom') ? 'bottom' :
                       modifiers.includes('left') ? 'left' :
                       modifiers.includes('right') ? 'right' : 'auto',
            trigger: modifiers.includes('click') ? 'click' :
                    modifiers.includes('focus') ? 'focus' :
                    modifiers.includes('manual') ? 'manual' : 'mouseenter',
            interactive: modifiers.includes('interactive'),
            allowHTML: modifiers.includes('html'),
            delay: modifiers.includes('delay') ? [500, 0] : 0,
        };
        
        effect(() => {
            const content = evaluate(expression);
            
            if (instance) {
                instance.setContent(content);
            } else {
                instance = tippy(el, {
                    content,
                    ...options
                });
            }
        });
        
        cleanup(() => {
            if (instance) {
                instance.destroy();
            }
        });
    });
    
    // Rich tooltip content builders
    Alpine.data('tooltipContent', () => ({
        episodeTooltip(episode) {
            return `
                <div class="text-sm">
                    <div class="font-semibold mb-1">Episode ${episode.episode_id}</div>
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div>Reward:</div>
                        <div class="text-right">${episode.total_reward.toFixed(1)}</div>
                        <div>Length:</div>
                        <div class="text-right">${episode.episode_length}</div>
                        <div>Reward Rate:</div>
                        <div class="text-right">${(episode.total_reward / episode.episode_length * 1000).toFixed(2)}/s</div>
                    </div>
                </div>
            `;
        },
        
        timelineTooltip(time, events) {
            const timeStr = this.formatTime(time);
            let content = `<div class="text-sm"><div class="font-semibold mb-1">Time: ${timeStr}</div>`;
            
            if (events && events.length > 0) {
                content += '<div class="mt-2 space-y-1">';
                events.forEach(event => {
                    const color = event.type === 'reward' ? 'text-yellow-400' :
                                 event.type === 'highValue' ? 'text-green-400' :
                                 'text-red-400';
                    content += `<div class="${color}">${event.type}: ${event.value.toFixed(2)}</div>`;
                });
                content += '</div>';
            }
            
            content += '</div>';
            return content;
        },
        
        statTooltip(label, description, trend) {
            return `
                <div class="text-sm">
                    <div class="font-semibold mb-1">${label}</div>
                    <div class="text-xs text-gray-400">${description}</div>
                    ${trend ? `<div class="mt-2 text-xs">Trend: ${trend}</div>` : ''}
                </div>
            `;
        },
        
        formatTime(ms) {
            const seconds = Math.floor(ms / 1000);
            const min = Math.floor(seconds / 60);
            const sec = seconds % 60;
            return `${min}:${sec.toString().padStart(2, '0')}`;
        }
    }));
    
    // Timeline tooltip manager
    Alpine.data('timelineTooltips', () => ({
        tooltipInstance: null,
        
        init() {
            // Create a single tooltip instance for the timeline
            this.tooltipInstance = tippy(this.$el, {
                content: 'Loading...',
                trigger: 'manual',
                followCursor: true,
                plugins: [followCursor],
                offset: [0, 10],
                arrow: false,
            });
        },
        
        showTooltip(e) {
            const rect = this.$el.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const progress = x / rect.width;
            const time = progress * this.store.playback.duration;
            
            // Find nearby events
            const events = this.findNearbyEvents(time);
            const content = this.$data.tooltipContent.timelineTooltip(time, events);
            
            this.tooltipInstance.setContent(content);
            this.tooltipInstance.show();
        },
        
        hideTooltip() {
            this.tooltipInstance.hide();
        },
        
        findNearbyEvents(time, threshold = 50) {
            // This would connect to the actual event data
            // For now, return mock data
            return [];
        }
    }));
    
    // Add chart tooltips
    Alpine.data('chartTooltips', () => ({
        init() {
            // Configure Chart.js to use Tippy for tooltips
            if (window.Chart) {
                Chart.defaults.plugins.tooltip.enabled = false;
                Chart.defaults.plugins.tooltip.external = function(context) {
                    const {chart, tooltip} = context;
                    
                    if (tooltip.opacity === 0) {
                        if (chart._tippyInstance) {
                            chart._tippyInstance.hide();
                        }
                        return;
                    }
                    
                    const content = tooltip.body.map(b => b.lines).flat().join('<br>');
                    
                    if (!chart._tippyInstance) {
                        chart._tippyInstance = tippy(chart.canvas, {
                            content,
                            trigger: 'manual',
                            placement: 'auto',
                            arrow: true,
                        });
                    } else {
                        chart._tippyInstance.setContent(content);
                    }
                    
                    chart._tippyInstance.setProps({
                        getReferenceClientRect: () => ({
                            left: tooltip.x,
                            top: tooltip.y,
                            right: tooltip.x,
                            bottom: tooltip.y,
                            width: 0,
                            height: 0,
                        }),
                    });
                    
                    chart._tippyInstance.show();
                };
            }
        }
    }));
});

// Import follow cursor plugin
const {followCursor} = tippy;