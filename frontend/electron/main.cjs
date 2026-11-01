const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

let mainWindow = null;
let backendProcess = null;
let ownsBackend = false;
let quitting = false;

const PORT = 8765;
const HEALTH_URL = `http://127.0.0.1:${PORT}/health`;

function checkHealth() {
  return new Promise((resolve) => {
    const request = http.get(HEALTH_URL, (response) => {
      let body = '';

      response.on('data', chunk => body += chunk);
      response.on('end', () => {
        try {
          const result = JSON.parse(body);
          resolve(
            response.statusCode === 200 &&
            result.application === 'SALAZAR' &&
            result.status === 'online'
          );
        } catch {
          resolve(false);
        }
      });
    });

    request.setTimeout(1000, () => request.destroy());
    request.on('error', () => resolve(false));
  });
}

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function startBackend() {
  if (!app.isPackaged) return true;

  // Reuse an existing SALAZAR backend if one is already running.
  if (await checkHealth()) return true;

  const executableName = process.platform === 'win32'
    ? 'salazar-backend.exe'
    : 'salazar-backend';

  const backendPath = path.join(
    process.resourcesPath, 'backend', executableName
  );

  const ragName = process.platform === 'win32'
    ? 'salazar-rag.exe'
    : 'salazar-rag';

  const ragPath = path.join(
    process.resourcesPath, 'rag', ragName
  );

  const root = process.env.SALAZAR_ROOT ||
    path.resolve(path.dirname(app.getPath('exe')), '../../..');

  backendProcess = spawn(backendPath, [], {
    cwd: root,
    windowsHide: true,
    stdio: 'ignore',
    env: {
      ...process.env,
      SALAZAR_ROOT: root,
      SALAZAR_PORT: String(PORT),
      SALAZAR_RAG_EXECUTABLE: ragPath
    }
  });

  ownsBackend = true;

  backendProcess.on('error', error => {
    console.error('SALAZAR backend launch error:', error);
  });

  for (let attempt = 0; attempt < 60; attempt++) {
    if (await checkHealth()) return true;

    if (backendProcess.exitCode !== null) break;

    await delay(500);
  }

  return false;
}

function stopBackend() {
  if (ownsBackend && backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }

  backendProcess = null;
  ownsBackend = false;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'SALAZAR',
    backgroundColor: '#09090b',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  if (process.env.SALAZAR_DEV_URL) {
    mainWindow.loadURL(process.env.SALAZAR_DEV_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  const ready = await startBackend();

  if (!ready) {
    dialog.showErrorBox(
      'SALAZAR Backend Error',
      'The SALAZAR backend could not start. Check that the application files are present on the USB drive.'
    );
    app.quit();
    return;
  }

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('before-quit', () => {
  quitting = true;
  stopBackend();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
