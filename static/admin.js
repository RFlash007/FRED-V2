(function() {
	const $ = (sel) => document.querySelector(sel);
	function secondsToHms(d) {
		d = Number(d);
		const h = Math.floor(d / 3600);
		const m = Math.floor((d % 3600) / 60);
		const s = Math.floor(d % 60);
		return [h, m, s]
			.map(v => String(v).padStart(2, '0'))
			.join(':');
	}

	async function fetchJSON(url) {
		const res = await fetch(url, { cache: 'no-store' });
		if (!res.ok) throw new Error('HTTP ' + res.status);
		return res.json();
	}

	function renderRecentConversation(list) {
		const container = $('#recent-conv');
		container.innerHTML = '';
		(list || []).forEach(turn => {
			const item = document.createElement('div');
			item.className = 'conv-item';
			const role = document.createElement('div');
			role.className = 'conv-role';
			role.textContent = (turn.role || 'unknown').toUpperCase();
			const text = document.createElement('div');
			text.className = 'conv-text';
			text.textContent = turn.content || '';
			item.appendChild(role);
			item.appendChild(text);
			container.appendChild(item);
		});
	}

	async function refreshOverview() {
		try {
			const data = await fetchJSON('/api/admin/overview');
			$('#pill-uptime').textContent = 'Uptime: ' + secondsToHms(data.uptime_sec || 0);
			$('#metric-ollama').textContent = (data.ollama && data.ollama.has_connection) ? 'Online' : 'Offline';
			$('#metric-webrtc').textContent = (data.webrtc && data.webrtc.bridge_online) ? 'Connected' : 'Disconnected';
			$('#metric-tts').textContent = data.tts && data.tts.available ? (data.tts.stewie_voice ? 'Ready (Voice Clone)' : 'Ready') : 'Unavailable';
			$('#metric-stt').textContent = data.stt && data.stt.is_initialized ? (data.stt.is_running ? 'Running' : 'Ready') : 'Unavailable';
			$('#metric-vision').textContent = data.vision && data.vision.available ? 'Available' : 'Unavailable';
			$('#vision-summary').textContent = (data.vision && data.vision.summary) || 'No visual context available.';
			$('#stat-messages').textContent = (data.conversation && data.conversation.current_messages) || 0;
			$('#stat-max').textContent = (data.conversation && data.conversation.max_messages) || 0;
			$('#stat-l2').textContent = (data.conversation && data.conversation.last_analyzed_index) || 0;
			$('#pill-turns').textContent = 'Turns: ' + ((data.conversation && data.conversation.total_turns) || 0);
			renderRecentConversation(data.recent_conversation || []);
		} catch (e) {
			console.error('overview error', e);
		}
	}

	async function refreshModels() {
		try {
			const data = await fetchJSON('/api/admin/models');
			$('#models-json').textContent = JSON.stringify(data, null, 2);
		} catch (e) {
			$('#models-json').textContent = '{"error":"failed to fetch model status"}';
		}
	}

	async function refreshMemoryStats() {
		try {
			const data = await fetchJSON('/api/admin/memory/stats');
			const container = document.querySelector('#memory-stats');
			if (!container) return;
			container.innerHTML = '';
			const section = document.createElement('div');
			section.className = 'memory-sections';
			// L2
			const l2 = data.l2 || {};
			const l2El = document.createElement('div');
			l2El.className = 'memory-card';
			l2El.innerHTML = `
				<div class="memory-card-header"><h4>L2 Episodic</h4><span class="pill">Realtime</span></div>
				<div class="memory-grid">
					<div><div class="metric-label">Total</div><div class="metric-value">${l2.total||0}</div></div>
					<div><div class="metric-label">Eligible for L3</div><div class="metric-value">${l2.eligible_for_consolidation||0}</div></div>
					<div><div class="metric-label">Last Created</div><div class="metric-value">${l2.last_created_at||'—'}</div></div>
				</div>
				<div class="subsection">
					<div class="sub-title">Recent</div>
					<div class="recent-list">${(l2.recent||[]).map(r=>`<div class="recent-row"><span>${r.created_at||''}</span><strong>${(r.topic||'').slice(0,80)}</strong><em>${r.turn_range||''}</em><code>${r.trigger_reason||''}</code></div>`).join('')}</div>
				</div>
				<div class="subsection">
					<div class="sub-title">Triggers (14d)</div>
					<div class="tag-list">${(l2.triggers||[]).map(t=>`<span class="tag">${t.reason}: ${t.count}</span>`).join('')}</div>
				</div>
			`;
			section.appendChild(l2El);
			// L3
			const l3 = data.l3 || {};
			const l3El = document.createElement('div');
			l3El.className = 'memory-card';
			l3El.innerHTML = `
				<div class="memory-card-header"><h4>L3 Long-Term</h4><span class="pill">Graph</span></div>
				<div class="memory-grid">
					<div><div class="metric-label">Nodes</div><div class="metric-value">${l3.nodes_total||0}</div></div>
					<div><div class="metric-label">Edges</div><div class="metric-value">${l3.edges_total||0}</div></div>
				</div>
				<div class="subsection">
					<div class="sub-title">Nodes by Type</div>
					<div class="tag-list">${(l3.nodes_by_type||[]).map(n=>`<span class="tag">${n.type}: ${n.count}</span>`).join('')}</div>
				</div>
				<div class="subsection">
					<div class="sub-title">Edges by Relationship</div>
					<div class="tag-list">${(l3.edges_by_rel||[]).map(e=>`<span class="tag">${e.rel_type}: ${e.count}</span>`).join('')}</div>
				</div>
				<div class="subsection">
					<div class="sub-title">Pending Edge Tasks</div>
					<div class="tag-list">${(l3.pending_tasks||[]).map(p=>`<span class="tag">${p.status}: ${p.count}</span>`).join('')}</div>
				</div>
				<div class="subsection">
					<div class="sub-title">Top Connected</div>
					<div class="recent-list">${(l3.top_connected||[]).map(n=>`<div class="recent-row"><strong>#${n.nodeid}</strong><span>${(n.label||'').slice(0,80)}</span><em>deg ${n.degree}</em></div>`).join('')}</div>
				</div>
			`;
			section.appendChild(l3El);
			container.appendChild(section);
		} catch (e) {
			console.error('memory stats error', e);
		}
	}

	document.addEventListener('DOMContentLoaded', () => {
		refreshOverview();
		refreshModels();
		refreshMemoryStats();
		setInterval(refreshOverview, 3000);
		setInterval(refreshMemoryStats, 5000);
		$('#btn-refresh-models').addEventListener('click', refreshModels);
	});
})();


