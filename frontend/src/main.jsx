import React, {useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import {
  Activity, AlertTriangle, CheckCircle2, Clock3, Database, Download,
  FileSearch, Fingerprint, Image as ImageIcon, Link2, Loader2, MapPin,
  Play, ShieldCheck, UploadCloud, Video, XCircle
} from "lucide-react";
import {jsPDF} from "jspdf";
import autoTable from "jspdf-autotable";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function Card({title, icon: Icon, children, className=""}) {
  return <section className={`card ${className}`}>
    <div className="card-title"><Icon size={18}/><span>{title}</span></div>
    {children}
  </section>
}

function Badge({children, tone="neutral"}) {
  return <span className={`badge ${tone}`}>{children}</span>
}

function scorePercent(result) {
  if (!result) return 0;
  const v = result.type === "Video"
    ? result.raw_verify_result?.video_result?.mean_score
    : result.raw_verify_result?.score;
  return Math.max(0, Math.min(100, Number(v || 0) * 100));
}

function verdictTone(v) {
  if (!v) return "neutral";
  if (v.includes("AI")) return "danger";
  if (v === "Authentic") return "success";
  return "warning";
}

function Dashboard() {
  const [file, setFile] = useState(null);
  const [source, setSource] = useState("Unknown");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");

  const percent = useMemo(() => scorePercent(result), [result]);
  const trace = result?.raw_trace_result;
  const verify = result?.raw_verify_result;

  async function analyze() {
    if (!file) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("source", source || "unknown");
      const res = await fetch(`${API}/api/analyze`, {method:"POST", body:fd});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.error || "Analysis failed");
      setResult(data);
      setTab("overview");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function downloadJSON() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {type:"application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href=url; a.download=`${result.id || "veritrace-case"}.json`; a.click();
    URL.revokeObjectURL(url);
  }

  function downloadPDF() {
    if (!result) return;
    const doc = new jsPDF();
    const margin = 14;
    doc.setFontSize(20); doc.text("VeriTrace Investigation Report", margin, 18);
    doc.setFontSize(9); doc.text(`Generated: ${new Date().toLocaleString()}`, margin, 25);
    doc.setFontSize(13); doc.text("Case Summary", margin, 38);
    autoTable(doc, {
      startY: 42,
      head: [["Field","Value"]],
      body: [
        ["Case ID", result.id],
        ["File", result.filename],
        ["Type", result.type],
        ["Source", result.source],
        ["Verdict", result.verdict],
        ["Confidence", `${result.confidence}%`],
        ["Model score", result.type==="Video" ? verify?.video_result?.mean_score : verify?.score],
        ["Threshold", verify?.threshold],
        ["SHA-256", result.hash],
        ["Provenance", result.provenance]
      ],
      styles:{fontSize:8}
    });

    let y = doc.lastAutoTable.finalY + 10;
    doc.setFontSize(13); doc.text("Tracing & Integrity", margin, y); y += 4;
    const summary = trace?.investigation_summary || {};
    autoTable(doc, {
      startY:y,
      head:[["Metric","Value"]],
      body:[
        ["Related records", summary.total_related_records ?? 0],
        ["Exact duplicates", summary.exact_duplicates ?? 0],
        ["Strong visual matches", summary.strong_visual_matches ?? 0],
        ["Possible matches", summary.possible_matches ?? 0],
        ["Distinct sources", summary.distinct_sources_observed ?? 0],
        ["Integrity", summary.integrity_status ?? "N/A"],
        ["External search", trace?.external_search?.status ?? "N/A"]
      ],
      styles:{fontSize:8}
    });

    y = doc.lastAutoTable.finalY + 10;
    const findings = result.investigation_findings || [];
    doc.setFontSize(13); doc.text("Investigation Findings", margin, y); y += 6;
    doc.setFontSize(9);
    findings.forEach((f,i) => {
      const lines = doc.splitTextToSize(`${i+1}. ${f}`, 180);
      doc.text(lines, margin, y); y += lines.length*4.5;
    });

    if (result.custody_entries?.length) {
      y += 4;
      doc.setFontSize(13); doc.text("Chain of Custody", margin, y); y += 4;
      autoTable(doc, {
        startY:y,
        head:[["Time","Actor","Action","Integrity"]],
        body:result.custody_entries.map(x=>[x.time,x.actor,x.action,"PASSED"]),
        styles:{fontSize:7}
      });
    }

    doc.save(`${result.id || "veritrace-report"}.pdf`);
  }

  return <div className="app">
    <header className="topbar">
      <div className="brand">
        <div className="logo"><ShieldCheck size={24}/></div>
        <div><b>VeriTrace</b><small>Evidence Intelligence Dashboard</small></div>
      </div>
      <div className="api-status"><span className="dot"></span> API {API}</div>
    </header>

    <main className="shell">
      <div className="hero">
        <div>
          <div className="eyebrow">DEEPFAKE + FORENSIC TRACE</div>
          <h1>Investigation Command Center</h1>
          <p>Upload an image or video and run detection, face analysis, provenance tracing, hashes, chain-of-custody and explainability from one dashboard.</p>
        </div>
        <div className="hero-chip"><Activity size={16}/> Unified analysis</div>
      </div>

      <div className="grid upload-grid">
        <Card title="Evidence Ingestion" icon={UploadCloud}>
          <label className="dropzone">
            <input type="file" accept="image/*,video/*" onChange={e=>setFile(e.target.files?.[0]||null)}/>
            <UploadCloud size={34}/>
            <b>{file ? file.name : "Drop evidence here or click to browse"}</b>
            <span>Images: JPG, PNG, WEBP, BMP · Video: MP4, MOV, AVI, MKV, WEBM</span>
          </label>
          <div className="form-row">
            <div>
              <label>Reported source</label>
              <input value={source} onChange={e=>setSource(e.target.value)} placeholder="e.g. WhatsApp, CCTV, Unknown"/>
            </div>
            <button className="primary" disabled={!file||loading} onClick={analyze}>
              {loading ? <><Loader2 className="spin" size={17}/> Processing...</> : <><Play size={17}/> Analyze Evidence</>}
            </button>
          </div>
          {error && <div className="error"><AlertTriangle size={17}/>{error}</div>}
        </Card>

        <Card title="Pipeline" icon={Activity}>
          <div className="pipeline">
            {["Upload","YuNet Face Detection","Xception Scoring","pHash + SHA-256","EXIF / GPS","Trace Timeline","Grad-CAM","Report PDF"].map((x,i)=>
              <div className={`pipe ${result ? "done":""}`} key={x}><span>{i+1}</span>{x}</div>
            )}
          </div>
          <p className="muted">The dashboard calls the unified <code>/api/analyze</code> endpoint, which already combines verification and tracing.</p>
        </Card>
      </div>

      {result && <div className="case">
        <div className="case-head">
          <div>
            <div className="eyebrow">CASE {result.id}</div>
            <h2>{result.filename}</h2>
            <div className="meta"><span><Clock3 size={14}/>{result.submitted}</span><span><Database size={14}/>{result.source}</span><span><Fingerprint size={14}/>{result.hash?.slice(0,18)}…</span></div>
          </div>
          <div className="actions">
            <button className="secondary" onClick={downloadJSON}><Download size={16}/> JSON</button>
            <button className="primary" onClick={downloadPDF}><Download size={16}/> Report PDF</button>
          </div>
        </div>

        <div className="tabs">
          {[
            ["overview","Overview"],["explain","Grad-CAM"],["trace","Trace & Provenance"],["custody","Custody"],["metadata","EXIF / Metadata"],["raw","Raw Output"]
          ].map(([k,v])=><button className={tab===k?"active":""} onClick={()=>setTab(k)} key={k}>{v}</button>)}
        </div>

        {tab==="overview" && <div className="grid dashboard-grid">
          <Card title="Verdict" icon={ShieldCheck} className="verdict-card">
            <div className="verdict-row">
              <div>
                <Badge tone={verdictTone(result.verdict)}>{result.verdict}</Badge>
                <div className="big-number">{result.confidence}%</div>
                <div className="muted">dashboard confidence</div>
              </div>
              <div className="gauge" style={{"--p":`${percent}%`}}><div>{Math.round(percent)}%</div></div>
            </div>
            <p>{result.summary}</p>
          </Card>

          <Card title="Detection Metrics" icon={FileSearch}>
            <div className="metric-list">
              <Metric label="Model score" value={result.type==="Video" ? verify?.video_result?.mean_score : verify?.score}/>
              <Metric label="Threshold" value={verify?.threshold}/>
              <Metric label="Frames analyzed" value={result.type==="Video" ? verify?.video_result?.frames_analyzed : 1}/>
              <Metric label="Max frame score" value={result.type==="Video" ? verify?.video_result?.max_score : verify?.score}/>
              <Metric label="Processing time" value={verify?.processing_time_sec ? `${Number(verify.processing_time_sec).toFixed(2)} sec` : "N/A"}/>
            </div>
          </Card>

          <Card title="Risk Alert" icon={AlertTriangle}>
            <Badge tone={result.risk_alert?.alert_level?.includes("NO MATCH") ? "success" : "danger"}>{result.risk_alert?.alert_level || "N/A"}</Badge>
            <p>{result.risk_alert?.message}</p>
            <div className="stat-strip"><b>{trace?.total_matches ?? 0}</b><span>related evidence records</span></div>
          </Card>

          <Card title="Propagation" icon={Link2}>
            {result.local_propagation_path?.length ? <div className="timeline">
              {result.local_propagation_path.map((x,i)=><div className="timeline-item" key={i}>
                <span className="timeline-dot"></span><div><b>{x.role}</b><small>{x.filename} · {x.source} · {x.uploaded_at}</small></div>
              </div>)}
            </div> : <Empty text="No prior propagation records found in the local evidence database."/>}
          </Card>
        </div>}

        {tab==="explain" && <Explain result={result}/>}
        {tab==="trace" && <Trace result={result}/>}
        {tab==="custody" && <Custody result={result}/>}
        {tab==="metadata" && <Metadata result={result}/>}
        {tab==="raw" && <pre className="json">{JSON.stringify(result,null,2)}</pre>}
      </div>}
    </main>
  </div>
}

function Metric({label,value}) {
  return <div className="metric"><span>{label}</span><b>{value ?? "N/A"}</b></div>
}
function Empty({text}) { return <div className="empty"><Database size={28}/><span>{text}</span></div> }

function Explain({result}) {
  const rawUrl =
    result?.gradcam_url ||
    result?.evidence?.gradcam_url ||
    result?.raw_verify_result?.gradcam_url ||
    result?.raw_verify_result?.evidence?.gradcam_url;

  const gradcamUrl = rawUrl
    ? rawUrl.startsWith("http://") || rawUrl.startsWith("https://")
      ? rawUrl
      : `${API}${rawUrl.startsWith("/") ? "" : "/"}${rawUrl}`
    : null;

  console.log("FULL RESULT:", result);
  console.log("Grad-CAM raw URL:", rawUrl);
  console.log("Grad-CAM final URL:", gradcamUrl);

  return (
    <div className="grid two">

      <Card title="Grad-CAM Explainability" icon={ImageIcon}>
        {gradcamUrl ? (
          <div>
            <img
              src={gradcamUrl}
              alt="Grad-CAM heatmap"
              className="heatmap"
              style={{
                width: "100%",
                maxWidth: "700px",
                display: "block",
                borderRadius: "12px"
              }}
              onLoad={() => console.log("Grad-CAM loaded:", gradcamUrl)}
              onError={(e) => {
                console.error("Grad-CAM failed to load:", gradcamUrl);
                console.error(e);
              }}
            />

            {result.type === "Video" &&
              result.gradcam_timestamp_sec != null && (
                <div className="muted" style={{marginTop: "10px"}}>
                  Heatmap generated around{" "}
                  <b>
                    {Number(result.gradcam_timestamp_sec).toFixed(2)}s
                  </b>

                  {result.gradcam_frame != null && (
                    <>
                      {" "}· Frame <b>{result.gradcam_frame}</b>
                    </>
                  )}
                </div>
              )}
          </div>
        ) : (
          <div className="empty">
            <AlertTriangle size={28}/>
            <span>
              Backend did not return a Grad-CAM image.
            </span>
          </div>
        )}
      </Card>

      <Card title="Interpretation" icon={FileSearch}>
        <div className="finding">
          <b>Why this result?</b>
          <p>
            The heatmap highlights regions contributing most strongly
            to the Xception prediction.
          </p>
        </div>

        <div className="finding">
          <b>Face pipeline</b>
          <p>
            YuNet detects faces first; the largest detected face is
            cropped and passed to Xception.
          </p>
        </div>

        {result.type === "Video" && (
          <div className="finding">
            <b>Video note</b>
            <p>
              The displayed Grad-CAM represents the sampled video frame
              selected by the backend.
            </p>
          </div>
        )}
      </Card>

    </div>
  );
}

function Trace({result}) {
  const s=result.raw_trace_result?.investigation_summary || {};
  return <div className="grid two">
    <Card title="Trace Summary" icon={Link2}>
      <div className="summary-grid">
        <Metric label="Related records" value={s.total_related_records}/>
        <Metric label="Exact duplicates" value={s.exact_duplicates}/>
        <Metric label="Strong visual" value={s.strong_visual_matches}/>
        <Metric label="Possible" value={s.possible_matches}/>
        <Metric label="Sources" value={s.distinct_sources_observed}/>
        <Metric label="Integrity" value={s.integrity_status}/>
      </div>
      <div className="notice"><b>External search:</b> {result.raw_trace_result?.external_search?.status}. {result.raw_trace_result?.external_search?.message}</div>
    </Card>
    <Card title="pHash Evidence" icon={Fingerprint}>
      {result.raw_trace_result?.timeline?.length ? <table><thead><tr><th>Filename</th><th>Source</th><th>Distance</th><th>Category</th></tr></thead><tbody>
        {result.raw_trace_result.timeline.map((m,i)=><tr key={i}><td>{m.filename}</td><td>{m.source}</td><td>{m.distance}</td><td><Badge tone={m.category==="CONFIRMED DUPLICATE"?"danger":"warning"}>{m.category}</Badge></td></tr>)}
      </tbody></table> : <Empty text="No related records."/>}
    </Card>
    <Card title="Investigation Findings" icon={FileSearch} className="span-2">
      <ul className="findings">{(result.investigation_findings||[]).map((f,i)=><li key={i}>{f}</li>)}</ul>
    </Card>
  </div>
}

function Custody({result}) {
  return <Card title="Chain of Custody" icon={ShieldCheck}>
    <table><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Detail</th><th>Integrity</th></tr></thead><tbody>
      {(result.custody_entries||[]).map((x,i)=><tr key={i}><td>{x.time}</td><td>{x.actor}</td><td>{x.action}</td><td>{x.detail}</td><td><Badge tone="success">PASSED</Badge></td></tr>)}
    </tbody></table>
  </Card>
}

function Metadata({result}) {
  const exif=result.exif_metadata || {};
  return <div className="grid two">
    <Card title="EXIF Availability" icon={MapPin}>
      <Badge tone={exif.available?"success":"warning"}>{exif.available?"AVAILABLE":"MISSING"}</Badge>
      <p>{exif.message || "Metadata extracted."}</p>
      {exif.gps_location && <div className="gps"><MapPin size={18}/><div><b>GPS</b><span>{exif.gps_location.latitude}, {exif.gps_location.longitude}</span><a href={exif.gps_location.google_maps_link} target="_blank">Open map</a></div></div>}
    </Card>
    <Card title="Metadata Fields" icon={Database}>
      {Object.keys(exif.metadata||{}).length ? <div className="metadata">{Object.entries(exif.metadata).map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div> : <Empty text="No readable EXIF fields."/>}
    </Card>
  </div>
}

createRoot(document.getElementById("root")).render(<Dashboard/>);
