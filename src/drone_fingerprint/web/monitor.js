'use strict';
const $ = id => document.getElementById(id);
let token = location.hash.slice(1) || sessionStorage.getItem('monitor-token') || '';
if (token) sessionStorage.setItem('monitor-token', token);
let selected = null, offset = 0, previous = null, samples = [], busy = false;
const statuses = ['completed', 'running', 'pending', 'failed', 'interrupted'];
const models = ['QDA', 'LDA', 'LSVM', 'RBF-SVM', 'KNN', 'DT', 'RF', 'GNB'];
const presets = ['figure2', 'figure3', 'figure4', 'table2', 'figure5', 'tables4_5'];
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function duration(seconds) {
  if (seconds === null || !Number.isFinite(seconds)) return '—';
  seconds = Math.max(0, Math.floor(seconds));
  return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds / 60) % 60}m ${seconds % 60}s`;
}
function secondsSince(start, end) { return start ? (Date.parse(end) - Date.parse(start)) / 1000 : null; }
async function api(path) {
  const response = await fetch(path, {headers: {Authorization: `Bearer ${token}`}, cache: 'no-store', signal: AbortSignal.timeout(8000)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
  return data;
}
function badge(status) { return node('span', status, `badge ${status}`); }
function chart(id, field, minimum) {
  const svg = $(id), ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const max = Math.max(minimum, ...samples.map(s => s[field] || 0));
  function add(tag, attributes, text) {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text) element.textContent = text;
    svg.append(element);
  }
  add('path', {d:'M 38 8 V 76 H 590', fill:'none', stroke:'#dce5e2'});
  add('text', {x:0, y:15}, max.toFixed(0)); add('text', {x:16, y:78}, '0');
  if (samples.length < 2) { add('text', {x:170, y:43}, 'Waiting for live process samples'); return; }
  const first = samples[0].at, last = samples.at(-1).at;
  let path = '', connected = false;
  for (const sample of samples) {
    if (sample[field] === null) { connected = false; continue; }
    const x = 38 + 552 * (sample.at - first) / Math.max(1, last - first);
    const y = 76 - 65 * sample[field] / max;
    path += `${connected ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)} `; connected = true;
  }
  add('path', {d:path, fill:'none', stroke:'#117864', 'stroke-width':2});
  add('text', {x:38, y:93}, new Date(first).toLocaleTimeString());
  add('text', {x:590, y:93, 'text-anchor':'end'}, new Date(last).toLocaleTimeString());
}
function renderCoverage(cells) {
  const table = $('coverage'); table.replaceChildren();
  const head = node('tr'); head.append(node('th', 'Model'));
  for (const preset of presets) head.append(node('th', preset.replace('figure', 'Fig. ').replace('tables4_5', 'Tbl. 4/5').replace('table2','Tbl. 2')));
  table.append(head);
  for (const model of models) {
    const row = node('tr'); row.append(node('th', model));
    for (const preset of presets) {
      const cell = cells.find(c => c.preset === preset && c.model === model);
      row.append(node('td', cell ? `${cell.completed}/${cell.total}` : '—', !cell ? '' : cell.failed ? 'problem' : cell.completed === cell.total ? 'done' : 'partial'));
    }
    table.append(row);
  }
}
function renderStatus(data) {
  $('run-name').textContent = data.name; $('run-path').textContent = data.output;
  $('run-status').textContent = data.status === 'unobserved' ? 'Runner not observed' : data.status;
  $('run-status').className = `badge ${data.status}`;
  const completed = data.counts.completed || 0;
  $('completed').textContent = `${completed.toLocaleString()} / ${data.total.toLocaleString()}`;
  $('completion-bar').value = data.total ? 100 * completed / data.total : 0;
  $('counts').textContent = statuses.filter(s => s !== 'completed').map(s => `${data.counts[s] || 0} ${s}`).join(' · ');
  $('elapsed').textContent = duration(secondsSince(data.started_at, data.finished_at || data.observed_at));
  let notice = 'Observing saved state. Task completion is not a runtime estimate; individual fits have no internal percentage or ETA.';
  if (data.status === 'waiting') notice = 'Waiting for plan.json and sweep.json. Start the paper command with this output directory; the monitor creates no run files.';
  if (data.status === 'unobserved') notice = 'The ledger says running, but no local runner lock is visible. The process may have stopped or be outside this PID/filesystem view. Inspect the terminal before resuming; do not assume it is still training.';
  if (data.status === 'completed') notice = 'The sweep ledger reports completion. Convergence warnings may still exist; use result export to verify scientific artifacts.';
  if (['failed', 'interrupted'].includes(data.status)) notice = 'The runner stopped before completing the plan. Inspect the attempt and log below. Resume is a separate terminal command; this page never starts or stops training.';
  $('notice').textContent = notice; $('notice').className = `notice ${['unobserved','failed','interrupted'].includes(data.status) ? 'warning' : ''}`;
  const detail = data.selected, progress = data.current_progress;
  const current = data.active_task && data.status === 'running';
  $('phase').textContent = current ? progress?.phase || 'Starting task' : data.status === 'running' ? 'Preparing / validating tasks' : 'No active phase';
  $('phase-time').textContent = current && progress?.status === 'running' ? `${duration(secondsSince(progress.at,data.observed_at))} in this phase` : 'No fit percentage or ETA is assumed.';
  const process = data.process, at = Date.parse(data.observed_at);
  let cpu = null;
  if (process && previous?.pid === process.pid && at > previous.at) cpu = Math.max(0,100000 * (process.cpu_seconds - previous.cpu_seconds) / (at - previous.at));
  previous = process ? {...process, at} : null;
  const rss = process ? process.rss_bytes / 1048576 : null;
  samples.push({at,cpu,rss}); samples = samples.filter(s => at-s.at <= 360000).slice(-120);
  $('resources').textContent = process ? `${cpu === null ? '…' : cpu.toFixed(0)+'%'} / ${rss.toFixed(0)} MiB` : '—';
  $('process-info').textContent = process ? `PID ${process.pid} · ${process.cpu_seconds.toFixed(1)} CPU seconds since process start` : 'No observable runner process';
  $('cpu-value').textContent = cpu === null ? '—' : `${cpu.toFixed(1)}%`;
  $('rss-value').textContent = rss === null ? '—' : `${rss.toFixed(1)} MiB`;
  $('io').textContent = process?.read_bytes !== undefined ? `Process storage I/O: ${(process.read_bytes/1048576).toFixed(1)} MiB read · ${(process.write_bytes/1048576).toFixed(1)} MiB written. These counters do not measure SSD wear.` : 'Storage I/O counters unavailable.';
  chart('cpu-chart','cpu',100); chart('rss-chart','rss',1); renderCoverage(data.coverage);
  $('selected-name').textContent = detail ? `${detail.preset} / ${detail.model} · ${detail.id}` : 'No task has started.';
  $('selected-state').textContent = detail ? `${selected ? 'Pinned task' : 'Following current / latest task'} · ${detail.status} · ${detail.frame_ms} ms · ${detail.n_filters} filters · ${detail.blocks} blocks · ${detail.portion}/3 coefficients` : '';
  $('task-error').textContent = detail?.error || '';
  $('attempts').replaceChildren();
  for (const [index, attempt] of (detail?.history || []).entries()) $('attempts').append(node('div', `Attempt ${(detail.attempts-detail.history.length)+index+1} · ${attempt.status} · ${attempt.started_at || '—'}`, 'attempt'));
  $('log').textContent = detail?.log.text || 'Waiting for the first attempt.';
  $('log-note').textContent = `${detail?.log.truncated ? 'Earlier log content omitted. ' : ''}Last 32 KiB only; complete logs remain on disk.${detail?.history_truncated ? ' Showing only the latest 50 attempts.' : ''}`;
  if ($('scroll-log').checked) $('log').scrollTop = $('log').scrollHeight;
  $('updated').textContent = `Last snapshot ${new Date(data.observed_at).toLocaleString()}`;
}
function renderTasks(data) {
  $('tasks').replaceChildren();
  for (const task of data.rows) {
    const row = node('tr'), title = node('td'), button = node('button', `${task.preset} / ${task.model}`, 'task-link');
    button.append(node('small',task.id)); button.onclick = () => { selected = task.id; refresh(); };
    title.append(button); row.append(title);
    row.append(node('td',`${task.frame_ms} ms / ${task.n_filters}`), node('td',`${task.blocks} / ${task.portion}/3`), node('td',`${task.snr_db === null ? 'Clean' : task.snr_db+' dB'} / ${task.trial ?? '—'}`));
    const state = node('td'); state.append(badge(task.status)); row.append(state,node('td',String(task.attempts))); $('tasks').append(row);
  }
  if (!data.rows.length) { const row=node('tr'), cell=node('td','No matching tasks'); cell.colSpan=6; row.append(cell); $('tasks').append(row); }
  $('task-range').textContent = data.total ? `${data.offset+1}–${Math.min(data.offset+50,data.total)} of ${data.total.toLocaleString()} tasks` : '0 tasks';
  $('previous').disabled = offset === 0; $('next').disabled = offset+50 >= data.total;
}
async function refresh() {
  if (busy) return; busy=true;
  try {
    const params = new URLSearchParams({status:$('status-filter').value, model:$('model-filter').value, search:$('search').value, offset});
    const [status,tasks] = await Promise.all([api('/api/status'+(selected ? '?task='+encodeURIComponent(selected) : '')),api('/api/tasks?'+params)]);
    renderStatus(status); renderTasks(tasks); $('connection').textContent='● Connected · 3s updates'; $('connection').className='connection live';
  } catch (error) {
    $('connection').textContent='● Disconnected / unreadable'; $('connection').className='connection';
    $('notice').textContent=`Live updates unavailable: ${error.message}. Any displayed values are from the last successful snapshot.`; $('notice').className='notice warning';
    previous=null; const at=Date.now();
    samples.push({at,cpu:null,rss:null});
    samples=samples.filter(s => at-s.at <= 360000).slice(-120);
  } finally { busy=false; }
}
for (const id of ['status-filter','model-filter']) $(id).onchange=()=>{offset=0;refresh();};
let debounce; $('search').oninput=()=>{clearTimeout(debounce);debounce=setTimeout(()=>{offset=0;refresh();},250);};
$('previous').onclick=()=>{offset=Math.max(0,offset-50);refresh();}; $('next').onclick=()=>{offset+=50;refresh();};
$('follow').onclick=()=>{selected=null;refresh();};
window.addEventListener('hashchange', () => {
  token = location.hash.slice(1);
  if (token) sessionStorage.setItem('monitor-token', token);
  else sessionStorage.removeItem('monitor-token');
  selected=null; offset=0; previous=null; samples=[]; refresh();
});
refresh(); setInterval(refresh,3000);
