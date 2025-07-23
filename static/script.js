/**
 * F.R.E.D. Enhanced Logging System - JavaScript Edition
 * Designed by Ollie-Tec™ - Post-Apocalyptic Computing Division
 * 
 * Browser-compatible logging with Stark Industries meets Vault-Tec theming
 */

function olliePrint(msg, level = 'info', module = null, showBanner = true, showComment = true) {
  // F.R.E.D. themed color scheme
  const colors = {
    info: 'color: #4FC3F7; font-weight: bold',        // Bright blue
    success: 'color: #4CAF50; font-weight: bold',     // Bright green  
    warning: 'color: #FFC107; font-weight: bold',     // Bright yellow
    error: 'color: #F44336; font-weight: bold',       // Bright red
    critical: 'color: white; background: #F44336; font-weight: bold', // White on red background
    debug: 'color: #00BCD4; font-weight: bold',       // Cyan
    audio: 'color: #E91E63; font-weight: bold',       // Bright magenta
    network: 'color: #00E5FF; font-weight: bold',     // Bright cyan
    optics: 'color: #00E5FF; font-weight: bold',      // Bright cyan
    armlink: 'color: #4CAF50; font-weight: bold',     // Bright green
    mainframe: 'color: #E91E63; font-weight: bold',   // Bright magenta
    shelter: 'color: #00E5FF; font-weight: bold',     // Bright cyan
    system: 'color: #FFFFFF; font-weight: normal'     // White
  };

  // Enhanced F.R.E.D. personality comments
  const comments = {
    info: [
      'Systems green across the board.',
      'All diagnostics nominal.',
      'Mainframe operating within parameters.',
      'ShelterNet protocols active.',
      'Data streams flowing smoothly.'
    ],
    success: [
      'Mission accomplished!',
      'Objective complete, field operative.',
      'Another successful operation.',
      'Target acquired and processed.',
      'Victory is ours today.'
    ],
    warning: [
      'Caution: power conduit unstable.',
      'Warning: anomalous readings detected.',
      'Attention: system irregularity observed.',
      'Alert: potential hazard identified.',
      'Advisory: proceed with enhanced vigilance.'
    ],
    error: [
      'Critical failure detected!',
      'System malfunction in progress.',
      'Emergency protocols activated.',
      'Catastrophic error encountered.',
      'Immediate intervention required.'
    ],
    critical: [
      '⚠️ VAULT BREACH IMMINENT ⚠️',
      '🚨 DEFCON 1 ACTIVATED 🚨',
      '💀 CORE MELTDOWN DETECTED 💀',
      '⛔ SYSTEM INTEGRITY COMPROMISED ⛔',
      '🔥 EMERGENCY SHUTDOWN REQUIRED 🔥'
    ],
    debug: [
      'Diagnostic mode active.',
      'Analyzing system matrices.',
      'Debug protocols engaged.',
      'Technical scan in progress.',
      'Detailed analysis available.'
    ],
    audio: [
      'Audio systems online.',
      'Voice synthesis ready.',
      'Communication channels clear.',
      'Sound processing active.',
      'Audio matrix stable.'
    ],
    network: [
      'Network protocols established.',
      'Communication links secured.',
      'Data transmission successful.',
      'Connection integrity verified.',
      'Network topology stable.'
    ],
    optics: [
      'Visual sensors calibrated.',
      'Camera systems operational.',
      'Image processing active.',
      'Optical analysis complete.',
      'Visual data acquired.'
    ],
    armlink: [
      'Field operative connection secure.',
      'ArmLink protocols established.',
      'Remote interface active.',
      'Field unit responding.',
      'Mobile operations nominal.'
    ],
    mainframe: [
      'Core intelligence systems online.',
      'Central processing unit active.',
      'Main server operational.',
      'Primary systems engaged.',
      'Control hub responding.'
    ],
    shelter: [
      'ShelterNet security active.',
      'Protected environment confirmed.',
      'Safe zone protocols enabled.',
      'Vault systems operational.',
      'Secure facility status green.'
    ]
  };

  // Auto-detect module name if not provided
  if (!module) {
    try {
      // Try to get calling function/file info from stack trace
      const stack = new Error().stack;
      const stackLines = stack.split('\n');
      const callerLine = stackLines[2] || stackLines[1] || '';
      const match = callerLine.match(/\/([^\/]+\.js)/);
      module = match ? match[1].replace('.js', '').toUpperCase() : 'BROWSER';
    } catch {
      module = 'BROWSER';
    }
  }

  // Get random comment for this level
  function getRandomComment(level) {
    const levelComments = comments[level] || comments.info;
    return levelComments[Math.floor(Math.random() * levelComments.length)];
  }

  // Create enhanced banner
  function createBanner(moduleName, level) {
    const timestamp = new Date().toISOString().substr(11, 8); // HH:MM:SS
    const ollieBrand = "OLLIE-TEC™";
    const techDivision = "Advanced Computing Division";
    const systemId = `Module: ${moduleName}`;
    const timestampStr = `Timestamp: ${timestamp}`;
    
    // Banner styling based on level
    let bannerStyle, accentStyle;
    if (['error', 'critical'].includes(level)) {
      bannerStyle = 'color: #F44336; font-weight: bold';
      accentStyle = 'color: white; background: #F44336; font-weight: bold';
    } else if (level === 'warning') {
      bannerStyle = 'color: #FFC107; font-weight: bold';
      accentStyle = 'color: #FFC107; font-weight: bold';
    } else if (level === 'success') {
      bannerStyle = 'color: #4CAF50; font-weight: bold';
      accentStyle = 'color: #4CAF50; font-weight: bold';
    } else {
      bannerStyle = 'color: #4FC3F7; font-weight: bold';
      accentStyle = 'color: #00E5FF; font-weight: bold';
    }

    // Build banner content
    const bannerWidth = 60;
    const borderLine = '═'.repeat(bannerWidth - 2);
    
    const banner = `
╔${borderLine}╗
║ ${ollieBrand.padStart((bannerWidth - 4 + ollieBrand.length) / 2).padEnd(bannerWidth - 4)} ║
║ ${techDivision.padStart((bannerWidth - 4 + techDivision.length) / 2).padEnd(bannerWidth - 4)} ║
║${'═'.repeat(bannerWidth - 2)}║
║ ${systemId.padEnd(bannerWidth - 4)} ║
║ ${timestampStr.padEnd(bannerWidth - 4)} ║
╚${borderLine}╝`.trim();

    console.log(`%c${banner}`, bannerStyle);
  }

  // Normalize level
  level = level.toLowerCase().trim();
  
  // Show banner if requested
  if (showBanner) {
    createBanner(module, level);
  }

  // Get styling for this level
  const style = colors[level] || colors.info;
  
  // Build the message
  let message = msg;
  if (showComment) {
    const comment = getRandomComment(level);
    const systemStyle = colors.system;
    console.log(`%c${message}%c → ${comment}`, style, systemStyle);
  } else {
    console.log(`%c${message}`, style);
  }
  
  // Add spacing for readability
  console.log('');
}

// === Specialized Logging Functions ===
function olliePrint_info(message, module = null) {
  olliePrint(message, 'info', module);
}

function olliePrint_success(message, module = null) {
  olliePrint(message, 'success', module);
}

function olliePrint_warning(message, module = null) {
  olliePrint(message, 'warning', module);
}

function olliePrint_error(message, module = null) {
  olliePrint(message, 'error', module);
}

function olliePrint_critical(message, module = null) {
  olliePrint(message, 'critical', module);
}

function olliePrint_debug(message, module = null) {
  olliePrint(message, 'debug', module);
}

// === Banner-only Functions ===
function olliePrint_simple(message, level = 'info') {
  olliePrint(message, level, null, false);
}

function olliePrint_quiet(message, level = 'info') {
  olliePrint(message, level, null, false, false);
}

// === Legacy Compatibility ===
// Keep existing function signatures for backward compatibility
window.olliePrint = olliePrint;
window.olliePrint_info = olliePrint_info;
window.olliePrint_success = olliePrint_success;
window.olliePrint_warning = olliePrint_warning;
window.olliePrint_error = olliePrint_error;
window.olliePrint_critical = olliePrint_critical;
window.olliePrint_debug = olliePrint_debug;
window.olliePrint_simple = olliePrint_simple;
window.olliePrint_quiet = olliePrint_quiet;

const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');
const chatHistory = document.getElementById('chat-history');
const modelSelect = document.getElementById('model-select');
const ollamaUrlInput = document.getElementById('ollama-url');
// const memoryTabsContainer = document.querySelector('.memory-tabs'); // REMOVED (Unused)
const memoryContentDisplay = document.getElementById('memory-content-display');
// const historyList = document.getElementById('history-list'); // Removed unused variable
const darkModeSwitch = document.getElementById('dark-mode-switch');
const neuralBackground = document.getElementById('fred-neural-bg');
const bodyElement = document.body;
const sidebarToggle = document.getElementById('sidebar-toggle');
// const idleModeSwitch = document.getElementById('idle-mode-switch'); // REMOVED (Feature removed)
const controlPanel = document.getElementById('control-panel');
const panelClose = document.getElementById('panel-close');

const activityLog = document.getElementById('activity-log'); // Get the activity log div

let isFredMuted = false; // Variable to track mute state

// --- Helper Functions ---

function logToolActivity(message) {
    if (!activityLog) return; // Safety check
    const activityEntry = document.createElement('p');
    activityEntry.textContent = message; // Use textContent for security
    activityLog.appendChild(activityEntry);
    // Scroll to the bottom of the activity log
    activityLog.scrollTop = activityLog.scrollHeight;
    
    // Limit activity log entries to prevent memory issues
    const entries = activityLog.children;
    if (entries.length > 50) {
        activityLog.removeChild(entries[0]);
    }
}

function appendMessage(role, content, addLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);
    if (addLoading && role === 'assistant') {
        messageDiv.classList.add('loading-placeholder');
    }

    // Create message structure
    const messageIndicator = document.createElement('div');
    messageIndicator.classList.add('message-indicator');
    
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    if (role === 'user') {
        avatar.classList.add('user-avatar');
        avatar.innerHTML = '<div class="avatar-core"></div>';
    } else if (role === 'assistant') {
        avatar.classList.add('system-avatar');
        avatar.innerHTML = '<div class="avatar-core"></div>';
    } else {
        avatar.classList.add('system-avatar');
        avatar.innerHTML = '<div class="avatar-core"></div>';
    }
    
    const messageMeta = document.createElement('div');
    messageMeta.classList.add('message-meta');
    
    const sender = document.createElement('span');
    sender.classList.add('sender');
    sender.textContent = role === 'user' ? 'USER' : role === 'assistant' ? 'F.R.E.D.' : 'SYSTEM';
    
    const timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    timestamp.textContent = new Date().toLocaleTimeString();
    
    messageMeta.appendChild(sender);
    messageMeta.appendChild(timestamp);
    messageIndicator.appendChild(avatar);
    messageIndicator.appendChild(messageMeta);
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    const paragraph = document.createElement('p');
    // Basic Markdown-like formatting (bold, italic) - could be expanded
    content = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>');       // Italic
    paragraph.innerHTML = content; // Use innerHTML for basic formatting
    messageContent.appendChild(paragraph);

    messageDiv.appendChild(messageIndicator);
    messageDiv.appendChild(messageContent);

    // Remove initial system message if it exists and this is the first *real* message
    const systemMessage = chatHistory.querySelector('.message.system-init');
    if (systemMessage && (role === 'user' || role === 'assistant')) {
        systemMessage.remove();
    }

    chatHistory.appendChild(messageDiv);
    // Smooth scroll to bottom
    chatHistory.scrollTo({
        top: chatHistory.scrollHeight,
        behavior: 'smooth'
    });
    
    // Trigger neural network activity
    if (window.neuralPulse && role === 'assistant') {
        window.neuralPulse(0.5);
    }
    
    return messageDiv;
}

// Make appendMessage globally available for the speech system
window.appendMessage = appendMessage;

function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto'; // Temporarily shrink
    const scrollHeight = textarea.scrollHeight;
    const maxHeight = parseInt(window.getComputedStyle(textarea).maxHeight, 10) || 120;

    if (scrollHeight > maxHeight) {
        textarea.style.height = maxHeight + 'px';
        textarea.style.overflowY = 'auto'; // Show scrollbar if max height reached
    } else {
        textarea.style.height = scrollHeight + 'px';
        textarea.style.overflowY = 'hidden'; // Hide scrollbar if not needed
    }
}

// --- Dark Mode ---
function applyDarkMode(isDark) {
    if (isDark) {
        bodyElement.classList.add('dark-mode');
    } else {
        bodyElement.classList.remove('dark-mode');
    }
}

function toggleDarkMode() {
    const isDarkMode = bodyElement.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
    darkModeSwitch.checked = isDarkMode;
}

// --- Control Panel Toggle ---
function toggleControlPanel() {
    const isOpen = bodyElement.classList.toggle('panel-open');
    controlPanel.classList.toggle('active', isOpen);
    localStorage.setItem('controlPanelOpen', isOpen);
    
    // Trigger resize to adjust mind map properly
    if (window.mindMap) {
        setTimeout(() => window.mindMap.resize(), 300); // Wait for transition to complete
    }
}

function closeControlPanel() {
    bodyElement.classList.remove('panel-open');
    controlPanel.classList.remove('active');
    localStorage.setItem('controlPanelOpen', false);
    
    if (window.mindMap) {
        setTimeout(() => window.mindMap.resize(), 300);
    }
}

// --- Status Updates ---
function updateSystemStatus(message, type = 'normal') {
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    
    if (statusText) {
        statusText.textContent = message;
    }
    
    if (statusDot) {
        statusDot.className = 'status-dot';
        switch (type) {
            case 'processing':
                statusDot.classList.add('processing');
                break;
            case 'error':
                statusDot.classList.add('error');
                break;
            case 'success':
                statusDot.classList.add('success');
                break;
            default:
                statusDot.classList.add('normal');
        }
    }
}

function updateMemoryDisplay(message) {
    if (memoryContentDisplay) {
        // Replace placeholder if it exists
        const placeholder = memoryContentDisplay.querySelector('.memory-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        const entry = document.createElement('div');
        entry.textContent = message;
        entry.style.marginBottom = '8px';
        entry.style.padding = '4px 8px';
        entry.style.background = 'rgba(var(--fred-focus-rgb), 0.05)';
        entry.style.borderRadius = '4px';
        entry.style.fontSize = '12px';
        entry.style.fontFamily = 'var(--font-mono)';
        
        memoryContentDisplay.appendChild(entry);
        
        // Limit entries
        const entries = memoryContentDisplay.children;
        if (entries.length > 10) {
            memoryContentDisplay.removeChild(entries[0]);
        }
    }
}

// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize textarea height
    adjustTextareaHeight(messageInput);
    
    // Check local storage for panel state
    const panelOpen = localStorage.getItem('controlPanelOpen') === 'true';
    if (panelOpen) {
        bodyElement.classList.add('panel-open');
        controlPanel.classList.add('active');
    }
    
    // Add click event listeners
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', toggleControlPanel);
    }
    
    if (panelClose) {
        panelClose.addEventListener('click', closeControlPanel);
    }
    
    // Close panel when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            controlPanel.classList.contains('active') &&
            !controlPanel.contains(e.target) &&
            !sidebarToggle.contains(e.target)) {
            closeControlPanel();
        }
    });
    
    // --- Mute Button Logic ---
    const muteFredButton = document.getElementById('muteFredButton');
    if (muteFredButton) {
        muteFredButton.addEventListener('click', () => {
            isFredMuted = !isFredMuted;
            muteFredButton.title = isFredMuted ? 'Unmute F.R.E.D.' : 'Mute F.R.E.D.';
            // Update button icon or style if needed
            if (isFredMuted) {
                muteFredButton.style.background = 'rgba(var(--fred-error), 0.2)';
            } else {
                muteFredButton.style.background = 'rgba(var(--fred-error), 0.1)';
            }
            olliePrint(`F.R.E.D. Muted: ${isFredMuted}`);
        });
    }
    
    // Initialize system status
    updateSystemStatus('Ready', 'success');
    
    olliePrint('F.R.E.D. Interface Initialized');
});

// Check local storage on load - Default to dark mode
const savedDarkMode = localStorage.getItem('darkMode');
const isDarkMode = savedDarkMode === null ? true : savedDarkMode === 'true'; // Default to true (dark mode)
applyDarkMode(isDarkMode);
darkModeSwitch.checked = isDarkMode;

// Save the default state if it's the first visit
if (savedDarkMode === null) {
    localStorage.setItem('darkMode', isDarkMode);
}

// Add listener to toggle
darkModeSwitch.addEventListener('change', toggleDarkMode);

// --- Event Listeners ---

messageForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default form submission

    const userMessage = messageInput.value.trim();
    if (!userMessage) return; // Don't send empty messages

    const selectedModel = modelSelect.value;
    const ollamaUrl = ollamaUrlInput.value.trim(); // Trim whitespace

    if (!ollamaUrl) {
        appendMessage('error', 'Server Endpoint URL cannot be empty.');
        return;
    }

    appendMessage('user', userMessage);
    messageInput.value = ''; // Clear input field
    adjustTextareaHeight(messageInput); // Reset textarea height
    messageInput.focus(); // Keep focus on input

    // Update system status
    updateSystemStatus('Processing...', 'processing');
    
    // Activate neural network
    if (window.neuralPulse) {
        window.neuralPulse(1);
    }
    
    // Start mind map animation
    if (window.mindMap) {
        window.mindMap.setActive(true);
    }

    const assistantMessageDiv = appendMessage('assistant', '', true); // Add loading placeholder
    const assistantParagraph = assistantMessageDiv.querySelector('p');
    let fullResponse = ""; // To accumulate the full response for potential Markdown parsing later

        // Build request payload, respecting backend default model
        const payload = {
            message: userMessage,
            ollama_url: ollamaUrl,
            mute_fred: isFredMuted // Send mute state to server
        };
        // Only include model if user explicitly selected a value
        if (selectedModel) {
            payload.model = selectedModel;
        }

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred, server response not JSON.' }));
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error || 'Server error'}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        assistantMessageDiv.classList.remove('loading-placeholder'); // Remove placeholder once stream starts
        updateSystemStatus('Receiving response...', 'processing');

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            // Process buffer line by line (assuming server sends JSON objects separated by newlines)
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
                const line = buffer.substring(0, newlineIndex).trim();
                buffer = buffer.substring(newlineIndex + 1);

                if (line) {
                    try {
                        const chunk = JSON.parse(line);
                        if (chunk.type === 'tool_activity' && chunk.content) {
                            logToolActivity(chunk.content);
                        } else if (chunk.response) {
                            fullResponse += chunk.response;
                            // Basic Markdown-like formatting for streaming
                            assistantParagraph.innerHTML = fullResponse
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/\n/g, '<br>'); // Replace explicit newlines if any
                        } else if (chunk.error) {
                            // Handle errors streamed from the backend
                            console.error("Streamed error:", chunk.error);
                            appendMessage('error', `Server error: ${chunk.error}`);
                        }
                    } catch (e) {
                        console.warn('Received non-JSON line or malformed JSON in stream:', line, e);
                        if (line.trim()) {
                             fullResponse += line;
                             assistantParagraph.innerHTML = fullResponse
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/\n/g, '<br>');
                        }
                    }
                }
                chatHistory.scrollTop = chatHistory.scrollHeight; // Keep scrolled to bottom
            }
        }
        
        // Final processing of any remaining buffer content
        if (buffer.trim()) {
            try {
                const chunk = JSON.parse(buffer.trim());
                 if (chunk.type === 'tool_activity' && chunk.content) {
                    logToolActivity(chunk.content);
                } else if (chunk.response) {
                    fullResponse += chunk.response;
                } else if (chunk.error) {
                    console.error("Streamed error (final buffer):", chunk.error);
                    appendMessage('error', `Server error: ${chunk.error}`);
                }
            } catch (e) {
                 console.warn('Received non-JSON final buffer content:', buffer.trim(), e);
                 fullResponse += buffer.trim();
            }
        }

        updateSystemStatus('Response complete', 'success');

    } catch (error) {
        console.error('Error sending message:', error);
        assistantParagraph.textContent = `Error: ${error.message}`;
        assistantMessageDiv.classList.remove('loading-placeholder');
        appendMessage('error', `Failed to get response from F.R.E.D.: ${error.message}`);
        updateSystemStatus('Error occurred', 'error');
    } finally {
        // Reset neural network activity
        if (window.neuralPulse) {
            setTimeout(() => {
                if (window.neuralNetwork) {
                    window.neuralNetwork.config.pulseSpeed = 0.02;
                    window.neuralNetwork.config.connectionOpacity = 0.15;
                }
            }, 1000);
        }
        
        if (window.mindMap) {
            window.mindMap.setActive(false);
        }
        
        // Reset status after delay
        setTimeout(() => {
            updateSystemStatus('Ready', 'success');
        }, 3000);
    }
});

// Textarea Input Handling (Auto-resize and Enter key)
messageInput.addEventListener('input', () => adjustTextareaHeight(messageInput));
messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        messageForm.requestSubmit();
    }
});

// Visualization controls
const vizReset = document.getElementById('viz-reset');
const vizCenter = document.getElementById('viz-center');

if (vizReset && window.mindMap) {
    vizReset.addEventListener('click', () => {
        window.mindMap.reset();
    });
}

if (vizCenter && window.mindMap) {
    vizCenter.addEventListener('click', () => {
        window.mindMap.center();
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    // Ctrl/Cmd + / to toggle control panel
    if ((event.ctrlKey || event.metaKey) && event.key === '/') {
        event.preventDefault();
        toggleControlPanel();
    }
    
    // Escape to close control panel
    if (event.key === 'Escape' && controlPanel.classList.contains('active')) {
        event.preventDefault();
        closeControlPanel();
    }
    
    // Ctrl/Cmd + D to toggle dark mode
    if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        toggleDarkMode();
    }
});

// Memory Tab Interaction
// memoryTabsContainer.addEventListener('click', (event) => {
//     if (event.target.classList.contains('memory-tab')) {
//         // Remove active class from all tabs
//         memoryTabsContainer.querySelectorAll('.memory-tab').forEach(tab => tab.classList.remove('active'));
//         // Add active class to the clicked tab
//         event.target.classList.add('active');
//         // Update placeholder content (replace with actual logic later)
//         const memoryType = event.target.dataset.memory;
//         // Add a temporary loading state visual
//         memoryContentDisplay.textContent = `Loading ${memoryType} view...`;
//         memoryContentDisplay.style.opacity = '0.5';
//         setTimeout(() => { // Simulate loading
//             memoryContentDisplay.textContent = `Accessing ${memoryType} memory...`;
//             memoryContentDisplay.style.opacity = '0.7';
//         }, 300);
//         olliePrint(`Switched to ${memoryType} memory view.`);

//     }
// });

// Initial Setup
adjustTextareaHeight(messageInput);
olliePrint('Fred UI Initialized');

// --- Potential Future Enhancements ---
// Additional features to implement:
// - Load/save chat history (localStorage or backend)
// - Dynamic model list population (requires backend endpoint)
// - Settings persistence (localStorage)
// - Implement actual memory content loading/display
// - More robust Markdown rendering

olliePrint('Clay UI Chat Initialized'); 
