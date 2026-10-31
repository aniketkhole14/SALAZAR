const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('salazar', {
  platform: process.platform,
  version: '1.0.0'
});
