/**
 * keywords: [polyfills, compatibility, browser]
 * 
 * Polyfills and compatibility fixes
 */

// Simple statistics global
if (typeof window.ss === 'undefined' && typeof simpleStatistics !== 'undefined') {
    window.ss = simpleStatistics;
}

// Ensure msgpack is available globally
if (typeof window.msgpack === 'undefined' && typeof msgpacklite !== 'undefined') {
    window.msgpack = msgpacklite;
}

// Mock LZ4 if not available
if (typeof window.LZ4 === 'undefined') {
    window.LZ4 = {
        decompress: function(data) {
            console.warn('LZ4 decompression not available, returning raw data');
            return data;
        }
    };
}