// Preload script for security isolation
const { contextBridge } = require('electron');

// Expose safe APIs to the renderer if needed
contextBridge.exposeInMainWorld('electronAPI', {
    platform: process.platform,
    version: process.versions.electron
});
