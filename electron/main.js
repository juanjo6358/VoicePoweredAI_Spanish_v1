const { app, BrowserWindow, Menu, session, globalShortcut } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

let mainWindow = null;
let pythonProcess = null;

const GRADIO_PORT = 7860;
const GRADIO_URL = `http://127.0.0.1:${GRADIO_PORT}`;

// Function to check if the port is ready
function waitForPort(port, host, timeout = 60000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    function tryConnect() {
      const socket = new net.Socket();

      socket.setTimeout(1000);
      socket.on('connect', () => {
        socket.destroy();
        resolve();
      });
      socket.on('error', () => {
        socket.destroy();
        if (Date.now() - startTime > timeout) {
          reject(new Error(`Timeout waiting for port ${port}`));
        } else {
          setTimeout(tryConnect, 500);
        }
      });
      socket.on('timeout', () => {
        socket.destroy();
        setTimeout(tryConnect, 500);
      });

      socket.connect(port, host);
    }

    tryConnect();
  });
}

// Set command line switches for audio recording and autoplay
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
app.commandLine.appendSwitch('use-fake-ui-for-media-stream'); // Auto-allows media streams

// Function to start the Python Gradio server
function startPythonServer() {
  const pythonPath = 'python3.12';
  const scriptPath = path.join(__dirname, '..', 'src', 'f5_tts', 'infer', 'infer_gradio.py');
  const srcPath = path.join(__dirname, '..', 'src');

  const env = { ...process.env };
  env.PYTHONPATH = srcPath + (env.PYTHONPATH ? `:${env.PYTHONPATH}` : '');
  env.PYTORCH_ENABLE_MPS_FALLBACK = '1'; // Fix for unsupported MPS operations on Mac (Intel & Silicon)

  console.log('Starting Python Gradio server...');
  pythonProcess = spawn(pythonPath, [scriptPath], {
    cwd: path.join(__dirname, '..'),
    env: env,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });

  return pythonProcess;
}

// Function to create the main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'VoicePowered AI',
    backgroundColor: '#1a1a2e',
    titleBarStyle: 'hiddenInset', // Modern Mac look
    trafficLightPosition: { x: 15, y: 15 },
    webPreferences: {
      nodeIntegration: true, // Enabled for better local compatibility
      contextIsolation: false, // Disabled to avoid issues with some Gradio JS libraries
      preload: path.join(__dirname, 'preload.js'),
      devTools: true,
      webSecurity: false, // Temporarily disabled to avoid file/media access issues
      sandbox: false // Disabled to prevent renderer crashes during hardware access
    },
    show: false // Don't show until ready
  });

  // Show window when ready to prevent flickering
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  return mainWindow;
}

// App startup
app.whenReady().then(async () => {
  // Hide default menu for cleaner look (optional)
  Menu.setApplicationMenu(null);

  // Configure media permissions for microphone access
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowedPermissions = ['media', 'microphone', 'audioCapture'];
    if (allowedPermissions.includes(permission)) {
      console.log(`Granting permission: ${permission}`);
      callback(true);
    } else {
      console.log(`Denying permission: ${permission}`);
      callback(false);
    }
  });

  // Also handle permission checks
  session.defaultSession.setPermissionCheckHandler((webContents, permission, requestingOrigin) => {
    const allowedPermissions = ['media', 'microphone', 'audioCapture'];
    if (allowedPermissions.includes(permission)) {
      return true;
    }
    return false;
  });

  // Register global shortcut for DevTools
  globalShortcut.register('CommandOrControl+Option+I', () => {
    if (mainWindow) mainWindow.webContents.openDevTools();
  });

  // Start Python server
  startPythonServer();

  // Create window
  const win = createWindow();

  // Handle crashes
  win.webContents.on('render-process-gone', (event, details) => {
    console.error('Render process gone:', details);
  });
  win.webContents.on('unresponsive', () => {
    console.warn('Window unresponsive');
  });

  // Show loading message
  win.loadURL(`data:text/html,
    <html>
      <head>
        <style>
          body {
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: white;
          }
          .container {
            text-align: center;
          }
          h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255,255,255,0.1);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
          }
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
          p {
            color: rgba(255,255,255,0.7);
            font-size: 1.1rem;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>VoicePowered AI</h1>
          <div class="spinner"></div>
          <p>Cargando el modelo de síntesis de voz...</p>
        </div>
      </body>
    </html>
  `);
  win.show();

  try {
    // Wait for Gradio server to be ready
    console.log('Waiting for Gradio server...');
    await waitForPort(GRADIO_PORT, '127.0.0.1', 120000); // 2 minute timeout for model loading
    console.log('Gradio server is ready!');

    // Load Gradio interface
    win.loadURL(GRADIO_URL);
  } catch (error) {
    console.error('Failed to connect to Gradio server:', error);
    win.loadURL(`data:text/html,
      <html>
        <head>
          <style>
            body {
              margin: 0;
              padding: 40px;
              display: flex;
              justify-content: center;
              align-items: center;
              height: 100vh;
              background: #1a1a2e;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              color: white;
              box-sizing: border-box;
            }
            .error {
              text-align: center;
              max-width: 600px;
            }
            h1 { color: #e74c3c; }
            p { color: rgba(255,255,255,0.7); line-height: 1.6; }
            code {
              background: rgba(255,255,255,0.1);
              padding: 2px 8px;
              border-radius: 4px;
            }
          </style>
        </head>
        <body>
          <div class="error">
            <h1>Error de Conexión</h1>
            <p>No se pudo conectar con el servidor de Gradio.</p>
            <p>Por favor, asegúrate de que Python 3.12 está instalado y que las dependencias están correctamente configuradas.</p>
          </div>
        </body>
      </html>
    `);
  }
});

app.on('window-all-closed', () => {
  // Kill Python process
  if (pythonProcess) {
    console.log('Terminating Python process...');
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    console.log('Terminating Python process before quit...');
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
