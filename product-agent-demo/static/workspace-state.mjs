export const clamp=(n,min,max)=>Math.max(min,Math.min(max,n));
export function normalizeRegion(a,b){const x=clamp(Math.min(a.x,b.x),0,1),y=clamp(Math.min(a.y,b.y),0,1);return{x,y,w:clamp(Math.max(a.x,b.x),0,1)-x,h:clamp(Math.max(a.y,b.y),0,1)-y};}
export function safeConfig(raw){return Object.fromEntries(Object.entries(raw).filter(([key])=>!key.endsWith('_key')));}
export function createSSEParser(receive){let buffer='';const consume=block=>{const data=block.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');if(data&&data!=='[DONE]')receive(JSON.parse(data));};const read=chunk=>{buffer+=chunk;let match;while((match=/\r?\n\r?\n/.exec(buffer))){const block=buffer.slice(0,match.index);buffer=buffer.slice(match.index+match[0].length);consume(block);}};read.flush=()=>{if(buffer.trim())consume(buffer);buffer='';};return read;}
export const agentDefinitions={
 facts:{name:'产品事实 Agent',icon:'file-text',description:'提取身份、规格、成分与用法'},
 review:{name:'声明核验 Agent',icon:'check-square',description:'核对包装声明与检索证据'},
 visual:{name:'图片核查 Agent',icon:'image',description:'检查包装文字与可见图片信息'}
};
export function emptyRun(){return {identity:null,input_type:'',batch_detected:false,sources:[],report:null,reports:{},agents:Object.fromEntries(Object.keys(agentDefinitions).map(id=>[id,{status:'idle',message:'等待共享识别与检索材料',draft:'',report:null}])),draft:'',error:'',elapsed:0,stages:{identify:{status:'idle',message:'等待图片识别'},search:{status:'idle',message:'识别产品后检索相关资料'},report:{status:'idle',message:'三个 Agent 等待共享材料'}}};}
export function batchProgress(items){const runs=items.map(item=>item?.run||emptyRun());const partialFailure=r=>!!r.error||Object.values(r.agents||{}).some(a=>a.status==='error');const finished=runs.filter(r=>['done','error','attention','cancelled'].includes(r.stages.report.status)||partialFailure(r)).length;const successful=runs.filter(r=>r.report&&r.stages.report.status==='done'&&!partialFailure(r)).length;return {total:runs.length,finished,successful,failed:Math.max(0,finished-successful)};}
export function reduceRun(previous,event){
 const r={...previous,reports:{...previous.reports},agents:structuredClone(previous.agents),stages:structuredClone(previous.stages)};
 const set=(k,status,message)=>r.stages[k]={status,message},id=event.model_id||'report';
 const agent=()=>r.agents[id]||(r.agents[id]={status:'idle',message:'',draft:'',report:null});
 switch(event.type){
  case 'started':set('identify','running',event.message);break;
  case 'product_identified':r.identity=event.product;set('identify','done','产品信息已提取');break;
  case 'input_classified':r.input_type=event.input_type||'';r.batch_detected=!!event.batch_detected;break;
  case 'needs_input':r.error=event.message;set('identify','attention',event.message);break;
  case 'search_started':set('search','running',event.message);break;
  case 'search_progress':if(r.stages.search.status==='running')r.stages.search.message=event.message;break;
  case 'sources':r.sources=event.sources||[];set('search','done',r.sources.length?`已返回 ${r.sources.length} 个资料来源`:'未检索到可追溯来源');break;
  case 'search_error':set('search','error',event.message);break;
  case 'model_started':
   if(r.stages.identify.status==='idle')set('identify','done','使用提供的产品名称');
   if(r.stages.search.status==='idle')set('search','skipped','本次未启用检索');
   set('report','running','三个 Agent 正在独立分析');
   Object.assign(agent(),{status:'running',message:'正在读取图片与共享资料',model:event.model});break;
  case 'token':agent().draft+=event.text||'';r.draft+=event.text||'';break;
  case 'model_retry':Object.assign(agent(),{status:'running',message:event.message||'报告格式异常，正在自动修复'});break;
  case 'report':{
   Object.assign(agent(),{status:'done',message:'分析完成',report:event.report});r.reports[id]=event.report;
   const reports=Object.values(r.reports);
   r.report={summary:reports.map(p=>`${p.agent_name||''}：${p.summary||''}`).join('\n'),official_facts:reports.flatMap(p=>p.official_facts||[]),claim_evidence_audit:reports.flatMap(p=>p.claim_evidence_audit||[]),image_observations:reports.flatMap(p=>p.image_observations||[]),evidence_gaps:[...new Set(reports.flatMap(p=>p.evidence_gaps||[]))],sources:r.sources};
   if(Object.values(r.agents).every(a=>['done','error','cancelled'].includes(a.status)))set('report','done','智能体分析已返回');
   break;
  }
  case 'model_error':Object.assign(agent(),{status:'error',message:event.message});r.error='部分智能体执行失败，可查看各自状态后重试';break;
  case 'error':r.error=event.message;for(const k of Object.keys(r.stages))if(r.stages[k].status==='running')set(k,'error',event.message);for(const a of Object.values(r.agents))if(a.status==='running'){a.status='error';a.message=event.message;}break;
  case 'cancelled':r.error='已取消本次分析';for(const k of Object.keys(r.stages))if(r.stages[k].status==='running')set(k,'cancelled','已取消');for(const a of Object.values(r.agents))if(a.status==='running'){a.status='cancelled';a.message='已取消';}break;
  case 'done':r.elapsed=event.elapsed_ms||0;set('report',r.report?'done':'error',r.report?'智能体结果已返回':'未返回有效分析结果');if(!r.report&&!r.error)r.error='三个 Agent 未返回有效分析结果';break;
 }
 return r;
}
