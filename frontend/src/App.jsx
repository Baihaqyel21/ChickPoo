import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  FileText,
  Gauge,
  ImagePlus,
  Leaf,
  Lightbulb,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Smartphone,
  Upload,
  Wheat,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const sections = [
  { id: 'scanner', label: 'Scanner', icon: ScanSearch },
  { id: 'model', label: 'Model', icon: BarChart3 },
  { id: 'guide', label: 'Panduan', icon: Lightbulb },
  { id: 'about', label: 'Project', icon: FileText },
];

const sampleImages = [
  { src: '/samples/healthy.jpg', label: 'Sehat', filename: 'healthy-sample.jpg' },
  { src: '/samples/cocci.jpg', label: 'Coccidiosis', filename: 'cocci-sample.jpg' },
  { src: '/samples/ncd.jpg', label: 'Newcastle', filename: 'ncd-sample.jpg' },
  { src: '/samples/salmo.jpg', label: 'Salmonella', filename: 'salmo-sample.jpg' },
];

const metrics = [
  { label: 'Accuracy', value: '98.34%', note: 'akurasi pada test set' },
  { label: 'Macro F1', value: '96.91%', note: 'metrik utama untuk dataset tidak seimbang' },
  { label: 'Recall NCD', value: '92.98%', note: 'kelas kritis tetap terbaca dengan baik' },
];

const classInfo = [
  { label: 'Sehat', key: 'healthy', text: 'Pola visual feses mendekati kondisi normal pada dataset.' },
  { label: 'Coccidiosis', key: 'cocci', text: 'Indikasi visual yang perlu dipantau karena berisiko mengganggu produktivitas.' },
  { label: 'Newcastle Disease', key: 'ncd', text: 'Kelas berisiko tinggi yang sebaiknya segera ditindaklanjuti.' },
  { label: 'Salmonella', key: 'salmo', text: 'Indikasi visual yang berkaitan dengan potensi gangguan pencernaan.' },
];

const guideItems = [
  'Dekatkan kamera sampai feses menjadi objek utama di dalam frame.',
  'Gunakan pencahayaan yang cukup agar warna dan tekstur terbaca jelas.',
  'Pastikan foto fokus, tidak buram, dan tidak terlalu gelap.',
  'Kurangi objek pengganggu seperti tangan, pakan, alas kandang yang terlalu ramai, atau tubuh ayam.',
  'Jika sistem meminta foto ulang, ambil gambar baru dari jarak dan cahaya yang lebih baik.',
];

const statusIcon = {
  accepted: CheckCircle2,
  review: AlertTriangle,
  needs_retake: ShieldAlert,
  invalid_input: ShieldAlert,
};

const sectionCopy = {
  scanner: {
    kicker: 'ChickPoo Scanner',
    title: 'Scanner kesehatan ayam',
    text: 'Unggah foto feses, ChickPoo memvalidasi objek lalu menampilkan prediksi, confidence, dan rekomendasi tindakan.',
  },
  model: {
    kicker: 'Model Terbaik',
    title: 'EfficientNetB0 frozen untuk klasifikasi empat kondisi',
    text: 'Model terbaik saat ini dipilih dari eksperimen training berdasarkan performa test set dan keseimbangan antar kelas.',
  },
  guide: {
    kicker: 'Panduan Input',
    title: 'Foto yang baik membuat hasil lebih dapat dipercaya',
    text: 'ChickPoo membaca citra. Semakin jelas objek feses di foto, semakin baik dasar sistem untuk melakukan screening.',
  },
  about: {
    kicker: 'Tentang Project',
    title: 'Machine learning yang dekat dengan kebutuhan peternak',
    text: 'ChickPoo dirancang sebagai alat screening awal yang mudah dijelaskan di kelas dan tetap mudah dipahami oleh peternak.',
  },
};

function formatPercent(value) {
  return `${Number(value || 0).toFixed(2)}%`;
}

function App() {
  const [entered, setEntered] = useState(false);
  const [activeSection, setActiveSection] = useState('scanner');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [insightOpen, setInsightOpen] = useState(true);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const activeCopy = sectionCopy[activeSection];
  const StatusIcon = useMemo(() => {
    if (!result) return ScanSearch;
    return statusIcon[result.status] || AlertTriangle;
  }, [result]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!cameraActive || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
    videoRef.current.play().catch(() => {
      setCameraError('Kamera belum bisa memutar preview. Coba tutup lalu buka lagi.');
    });
  }, [cameraActive]);

  function enterApp() {
    setEntered(true);
    window.setTimeout(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 0);
  }

  function changeSection(sectionId) {
    setActiveSection(sectionId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function updateSelectedFile(file) {
    setError('');
    setResult(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function handleFileInput(event) {
    const file = event.target.files?.[0];
    if (file) updateSelectedFile(file);
    event.target.value = '';
  }

  async function useSampleImage(item) {
    const response = await fetch(item.src);
    const blob = await response.blob();
    updateSelectedFile(new File([blob], item.filename, { type: blob.type || 'image/jpeg' }));
    changeSection('scanner');
  }

  async function startCamera() {
    setCameraError('');
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch {
      setCameraError('Kamera tidak bisa diakses. Periksa izin kamera di browser.');
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setCameraError('Kamera belum siap. Tunggu sebentar lalu coba lagi.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `chickpoo-camera-${Date.now()}.jpg`, { type: 'image/jpeg' });
      updateSelectedFile(file);
      stopCamera();
    }, 'image/jpeg', 0.92);
  }

  async function submitPrediction() {
    if (!selectedFile) {
      setError('Pilih atau ambil foto terlebih dahulu.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Prediksi gagal diproses.');
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || 'Backend belum siap atau tidak bisa dihubungi.');
    } finally {
      setIsLoading(false);
    }
  }

  function resetInput() {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setPreviewUrl('');
  }

  if (!entered) {
    return <LandingGate onEnter={enterApp} />;
  }

  return (
    <main className={`app-shell ${insightOpen ? 'insight-open' : 'insight-closed'}`}>
      <header className="app-topbar">
        <div className="topbar-inner">
          <button className="brand-lockup" type="button" onClick={() => changeSection('scanner')} title="Ke scanner">
            <span className="brand-mark" aria-hidden="true">
              <Leaf size={24} />
            </span>
            <span className="brand-copy">
              <strong>ChickPoo</strong>
              <small>AI screening feses ayam</small>
            </span>
          </button>

          <nav className="top-nav" aria-label="Navigasi utama">
            {sections.map((section) => {
              const Icon = section.icon;
              return (
                <button
                  className={activeSection === section.id ? 'active' : ''}
                  type="button"
                  key={section.id}
                  onClick={() => changeSection(section.id)}
                >
                  <Icon size={18} />
                  <span>{section.label}</span>
                </button>
              );
            })}
          </nav>

          <button className="insight-toggle" type="button" onClick={() => setInsightOpen((value) => !value)}>
            <SlidersHorizontal size={18} />
            <span>{insightOpen ? 'Tutup panel' : 'Buka panel'}</span>
          </button>
        </div>
      </header>

      <div className="main-stage">
        <section className="app-header">
          <div className="header-copy">
            <p className="eyebrow">{activeCopy.kicker}</p>
            <h1>{activeCopy.title}</h1>
            <p>{activeCopy.text}</p>
          </div>
          <div className="header-signal">
            <span>
              <ShieldCheck size={18} />
              Validity gate aktif
            </span>
            <strong>Quality + visual + embedding</strong>
            <small>Menahan input non-feses sebelum prediksi penyakit ditampilkan.</small>
          </div>
        </section>

        {activeSection === 'scanner' && (
          <ScannerView
            insightOpen={insightOpen}
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            cameraActive={cameraActive}
            cameraError={cameraError}
            isLoading={isLoading}
            result={result}
            error={error}
            videoRef={videoRef}
            StatusIcon={StatusIcon}
            handleFileInput={handleFileInput}
            startCamera={startCamera}
            stopCamera={stopCamera}
            capturePhoto={capturePhoto}
            submitPrediction={submitPrediction}
            resetInput={resetInput}
            useSampleImage={useSampleImage}
          />
        )}
        {activeSection === 'model' && <ModelView changeSection={changeSection} />}
        {activeSection === 'guide' && <GuideView changeSection={changeSection} />}
        {activeSection === 'about' && <AboutView changeSection={changeSection} />}
      </div>
    </main>
  );
}

function LandingGate({ onEnter }) {
  return (
    <main className="landing-gate">
      <div className="landing-shade" />
      <section className="landing-copy" aria-label="Pembuka ChickPoo">
        <p className="landing-kicker">AI screening untuk peternakan ayam modern</p>
        <h1>ChickPoo</h1>
        <p>
          Prediksi awal penyakit ayam dari foto feses, tampilkan tingkat keyakinan model, lalu
          berikan rekomendasi tindakan yang jelas untuk langkah berikutnya.
        </p>
        <button className="landing-cta" type="button" onClick={onEnter}>
          <ScanSearch size={22} />
          <span>Mulai Deteksi</span>
          <ArrowRight size={20} />
        </button>
        <small>Dirancang untuk pembelajaran machine learning dan screening awal di lapangan.</small>
      </section>
    </main>
  );
}

function ScannerView({
  insightOpen,
  selectedFile,
  previewUrl,
  cameraActive,
  cameraError,
  isLoading,
  result,
  error,
  videoRef,
  StatusIcon,
  handleFileInput,
  startCamera,
  stopCamera,
  capturePhoto,
  submitPrediction,
  resetInput,
  useSampleImage,
}) {
  return (
    <section className={`scanner-view page-motion ${insightOpen ? '' : 'insight-closed'}`}>
      <div className="scanner-console">
        <section className="capture-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Input Gambar</p>
              <h2>Ambil foto feses ayam</h2>
            </div>
            {selectedFile && (
              <button className="icon-button" type="button" onClick={resetInput} title="Reset foto">
                <RotateCcw size={19} />
              </button>
            )}
          </div>

          <div className="preview-stage">
            {cameraActive ? (
              <video ref={videoRef} className="camera-view" muted playsInline />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Preview foto yang akan diprediksi" />
            ) : (
              <div className="empty-preview">
                <ImagePlus size={50} />
                <span>Foto belum dipilih</span>
              </div>
            )}
          </div>

          <div className="capture-controls">
            <label className="control-button" title="Unggah gambar">
              <Upload size={19} />
              <span>Unggah</span>
              <input type="file" accept="image/*" onChange={handleFileInput} />
            </label>
            {cameraActive ? (
              <>
                <button className="control-button primary" type="button" onClick={capturePhoto} title="Ambil foto">
                  <Camera size={19} />
                  <span>Ambil</span>
                </button>
                <button className="control-button neutral" type="button" onClick={stopCamera} title="Tutup kamera">
                  <X size={19} />
                  <span>Tutup</span>
                </button>
              </>
            ) : (
              <button className="control-button" type="button" onClick={startCamera} title="Buka kamera">
                <Camera size={19} />
                <span>Kamera</span>
              </button>
            )}
          </div>

          {cameraError && <p className="inline-alert">{cameraError}</p>}
          {error && <p className="inline-alert strong">{error}</p>}

          <button
            className="scan-button"
            type="button"
            onClick={submitPrediction}
            disabled={isLoading || !selectedFile}
            title="Jalankan prediksi"
          >
            {isLoading ? <LoaderCircle className="spin" size={21} /> : <ScanSearch size={21} />}
            <span>{isLoading ? 'Menganalisis foto...' : 'Prediksi Penyakit Ayam'}</span>
          </button>
        </section>

        <section className={`diagnosis-panel ${result ? 'has-result' : ''}`}>
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Hasil Prediksi</p>
              <h2>
                {result?.status === 'invalid_input'
                  ? 'Input perlu foto ulang'
                  : result
                    ? 'Diagnosis awal siap dibaca'
                    : 'Belum ada hasil'}
              </h2>
            </div>
            <span className={`status-pill ${result?.status || 'idle'}`}>
              <StatusIcon size={18} />
              <span>{result ? result.recommendation.confidence_level : 'siap'}</span>
            </span>
          </div>

          {result ? (
            <ResultContent result={result} />
          ) : (
            <div className="diagnosis-empty">
              <Activity size={56} />
              <h3>Masukkan foto untuk memulai screening</h3>
              <p>
                Setelah diproses, ChickPoo akan menampilkan prediksi penyakit ayam, persentase
                keyakinan model, transparansi tiap kelas, dan rekomendasi tindakan.
              </p>
            </div>
          )}
        </section>

        {insightOpen && <InsightPanel result={result} />}

        <SampleTray useSampleImage={useSampleImage} />
      </div>
    </section>
  );
}

function InsightPanel({ result }) {
  return (
    <aside className="insight-panel">
      <section className="insight-section">
        <p className="eyebrow">Validasi Input</p>
        <h2>Foto harus menyerupai pola feses ayam</h2>
        <p>
          Input diperiksa dari kualitas foto, pola warna-tekstur, dan kemiripan embedding terhadap
          distribusi dataset feses ayam.
        </p>
      </section>

      <section className="insight-section">
        <p className="eyebrow">Status</p>
        <h2>{result ? result.recommendation.confidence_level : 'Siap memindai'}</h2>
        <p>
          {result
            ? result.status_message
            : 'Masukkan foto feses ayam yang fokus dan cukup terang untuk memulai screening.'}
        </p>
      </section>
    </aside>
  );
}

function SampleTray({ useSampleImage }) {
  return (
    <section className="sample-tray" aria-label="Contoh dataset ChickPoo">
      <div className="sample-tray-copy">
        <p className="eyebrow">Demo Dataset</p>
        <h2>Contoh cepat</h2>
      </div>
      <div className="sample-list">
        {sampleImages.map((item) => (
          <button type="button" key={item.label} onClick={() => useSampleImage(item)} title={`Pakai contoh ${item.label}`}>
            <img src={item.src} alt={`Contoh ${item.label}`} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function ResultContent({ result }) {
  const isInvalidInput = result.status === 'invalid_input';
  const relevanceNotes = result.input_assessment?.flags || [];

  return (
    <>
      <div className={`diagnosis-spotlight ${isInvalidInput ? 'invalid' : ''}`}>
        <span>{isInvalidInput ? 'Status input' : 'Prediksi penyakit ayam'}</span>
        <h3>{result.predicted_label}</h3>
        {isInvalidInput ? (
          <div className="confidence-line">
            <ShieldAlert size={21} />
            <strong>Foto ulang</strong>
            <em>objek tidak sesuai</em>
          </div>
        ) : (
          <div className="confidence-line">
            <Gauge size={21} />
            <strong>{formatPercent(result.confidence_percentage)}</strong>
            <em>keyakinan model</em>
          </div>
        )}
        <p>{result.status_message}</p>
      </div>

      {isInvalidInput ? (
        <article className="invalid-input-box">
          <div className="subheading">
            <p className="eyebrow">Validasi Objek</p>
            <h3>Prediksi penyakit ditahan</h3>
          </div>
          <p>
            Model klasifikasi tetap menghasilkan angka untuk setiap gambar, tetapi ChickPoo tidak
            menampilkannya sebagai diagnosis ketika foto terindikasi bukan feses ayam.
          </p>
          {relevanceNotes.length > 0 && (
            <ul>
              {relevanceNotes.map((note) => (
                <li key={note.code}>{note.message}</li>
              ))}
            </ul>
          )}
        </article>
      ) : (
        <div className="probability-section">
          <div className="subheading">
            <p className="eyebrow">Transparansi</p>
            <h3>Persentase tiap kelas</h3>
          </div>
          <div className="probability-list">
            {result.probabilities.map((item) => (
              <div className="probability-row" key={item.key}>
                <div className="probability-meta">
                  <span>{item.label}</span>
                  <strong>{formatPercent(item.percentage)}</strong>
                </div>
                <div className="bar-track" aria-hidden="true">
                  <span style={{ width: `${Math.max(item.percentage, 1)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <article className="recommendation-box">
        <div className="subheading">
          <p className="eyebrow">Rekomendasi</p>
          <h3>{result.recommendation.headline}</h3>
        </div>
        <p>{result.recommendation.confidence_message}</p>
        <ol>
          {result.recommendation.actions.map((action) => (
            <li key={action}>{action}</li>
          ))}
        </ol>
      </article>

      {result.recommendation.photo_notes.length > 0 && (
        <article className="photo-notes">
          <div className="subheading">
            <p className="eyebrow">Catatan Foto</p>
            <h3>Kualitas input</h3>
          </div>
          <ul>
            {result.recommendation.photo_notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </article>
      )}

      <p className="disclaimer">{result.recommendation.disclaimer}</p>
    </>
  );
}

function ModelView({ changeSection }) {
  return (
    <section className="model-view page-motion">
      <div className="metrics-grid">
        {metrics.map((item) => (
          <article key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <p>{item.note}</p>
          </article>
        ))}
      </div>

      <div className="model-layout">
        <figure className="evidence-figure">
          <img src="/assets/confusion-matrix.png" alt="Confusion matrix model EfficientNetB0 frozen" />
          <figcaption>Confusion matrix test set untuk model EfficientNetB0 frozen.</figcaption>
        </figure>

        <section className="class-panel">
          <p className="eyebrow">Kelas Model</p>
          <h2>Empat kondisi yang dikenali</h2>
          <div className="class-list">
            {classInfo.map((item) => (
              <article key={item.key}>
                <strong>{item.label}</strong>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
          <button className="primary-action compact" type="button" onClick={() => changeSection('scanner')}>
            <ScanSearch size={19} />
            <span>Coba Scanner</span>
            <ChevronRight size={18} />
          </button>
        </section>
      </div>
    </section>
  );
}

function GuideView({ changeSection }) {
  return (
    <section className="guide-view page-motion">
      <div className="guide-layout">
        <article className="guide-feature">
          <Smartphone size={32} />
          <h2>Prinsip utama</h2>
          <p>
            Foto terbaik adalah foto yang membuat feses menjadi objek utama, cukup terang, fokus,
            dan tidak tertutup objek lain.
          </p>
          <button className="primary-action compact" type="button" onClick={() => changeSection('scanner')}>
            <Camera size={19} />
            <span>Buka Scanner</span>
          </button>
        </article>

        <article className="guide-checklist">
          <ClipboardList size={32} />
          <h2>Checklist foto</h2>
          <ol>
            {guideItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ol>
        </article>

        <article className="guide-warning">
          <ShieldAlert size={32} />
          <h2>Jika objek bukan feses ayam</h2>
          <p>
            Sistem memakai quality check, sinyal warna-tekstur, dan pembanding embedding terhadap
            dataset feses ayam. Jika pola visual berada di luar distribusi tersebut, ChickPoo akan
            menahan prediksi penyakit dan meminta foto ulang.
          </p>
        </article>
      </div>
    </section>
  );
}

function AboutView({ changeSection }) {
  return (
    <section className="about-view page-motion">
      <div className="about-grid">
        <article>
          <Wheat size={30} />
          <h2>Masalah lapangan</h2>
          <p>
            Gejala penyakit ayam sering terlambat dikenali. Untuk peternak kecil, keterlambatan
            ini bisa berujung pada kerugian ekonomi yang besar.
          </p>
        </article>
        <article>
          <ScanSearch size={30} />
          <h2>Solusi MVP</h2>
          <p>
            ChickPoo membantu melakukan screening awal melalui foto feses, lalu menyajikan hasil
            dalam bahasa yang mudah dibaca.
          </p>
        </article>
        <article>
          <ShieldCheck size={30} />
          <h2>Batasan etis</h2>
          <p>
            Hasil aplikasi bukan diagnosis final. Pengguna tetap perlu memeriksa kondisi ayam dan
            berkonsultasi dengan petugas kesehatan hewan.
          </p>
        </article>
      </div>

      <section className="closing-band">
        <p className="eyebrow">ChickPoo</p>
        <h2>Lebih cepat membaca tanda awal, lebih siap mengambil tindakan.</h2>
        <button className="primary-action compact" type="button" onClick={() => changeSection('scanner')}>
          <ArrowRight size={19} />
          <span>Mulai Deteksi</span>
        </button>
      </section>
    </section>
  );
}

export default App;
