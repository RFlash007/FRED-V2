function olliePrint(msg, level = 'info') {
  const colors = {info: '\x1b[34m', success: '\x1b[32m', warning: '\x1b[33m', error: '\x1b[31m'};
  const reset = '\x1b[0m';
  const comments = {info: 'Systems green across the board.', success: 'Mission accomplished!', warning: 'Caution: power conduit unstable.', error: 'Critical failure detected!'};
  const ts = new Date().toISOString();
  const header = `BROWSER [${ts}] - Designed by Ollie-Tec`;
  const bar = '-'.repeat(header.length);
  console.log(`${bar}\n${header}\n${bar}\n${colors[level] || ''}${msg}${reset} ${comments[level] || ''}`);
}

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

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: userMessage,
                model: selectedModel,
                ollama_url: ollamaUrl,
                mute_fred: isFredMuted // Send mute state to server
            }),
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
