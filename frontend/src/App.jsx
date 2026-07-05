import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  Gauge,
  ImagePlus,
  LoaderCircle,
  Menu,
  RotateCcw,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Moon,
  Sparkles,
  Sun,
  Upload,
  Wheat,
  X,
} from 'lucide-react';
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const navItems = [
  { id: 'home', label: 'Beranda' },
  { id: 'scanner', label: 'Scanner' },
  { id: 'workflow', label: 'Cara Kerja' },
  { id: 'model', label: 'Model' },
];

const sampleImages = [
  { src: '/samples/healthy.jpg', label: 'Sehat', filename: 'healthy-sample.jpg' },
  { src: '/samples/cocci.jpg', label: 'Coccidiosis', filename: 'cocci-sample.jpg' },
  { src: '/samples/ncd.jpg', label: 'Newcastle', filename: 'ncd-sample.jpg' },
  { src: '/samples/salmo.jpg', label: 'Salmonella', filename: 'salmo-sample.jpg' },
];

const modelMetrics = [
  { label: 'Object Gate Accuracy', value: '99.58%', note: 'Akurasi validasi feses ayam vs objek lain.', Icon: ShieldCheck },
  { label: 'Object Gate F1', value: '99.61%', note: 'Keseimbangan ketepatan dan sensitivitas validasi objek.', Icon: CheckCircle2 },
  { label: 'Disease Classifier Accuracy', value: '97.85%', note: 'Akurasi klasifikasi kondisi kesehatan pada data test.', Icon: BarChart3 },
  { label: 'Macro F1', value: '97.38%', note: 'Keseimbangan performa antar kelas penyakit.', Icon: Gauge },
];

const classInfo = [
  { label: 'Sehat', key: 'healthy', tone: 'good', text: 'Pola visual mendekati kondisi normal.' },
  { label: 'Coccidiosis', key: 'cocci', tone: 'amber', text: 'Perlu pemantauan karena dapat mengganggu produktivitas.' },
  { label: 'Salmonella', key: 'salmo', tone: 'orange', text: 'Berkaitan dengan potensi gangguan pencernaan dan sanitasi.' },
  { label: 'Newcastle Disease', key: 'ncd', tone: 'danger', text: 'Kelas berisiko tinggi dan perlu tindak lanjut cepat.' },
];

const statusIcon = {
  accepted: CheckCircle2,
  review: AlertTriangle,
  needs_retake: ShieldAlert,
  invalid_input: ShieldAlert,
};

const confusionMatrices = [
  {
    title: 'Object Gate Confusion Matrix',
    labels: ['not_chicken_feces', 'chicken_feces'],
    data: [
      [501, 1],
      [6, 1016],
    ],
  },
  {
    title: 'Disease Classifier Confusion Matrix',
    labels: ['cocci', 'healthy', 'ncd', 'salmo'],
    data: [
      [313, 0, 3, 0],
      [0, 303, 2, 3],
      [0, 3, 53, 1],
      [2, 2, 1, 336],
    ],
  },
];

function getInitialTheme() {
  if (typeof window === 'undefined') return 'dark';
  try {
    const savedTheme = window.localStorage.getItem('theme');
    if (savedTheme === 'dark' || savedTheme === 'light') return savedTheme;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  } catch {
    return 'dark';
  }
}

function formatPercent(value, maximumFractionDigits = 1) {
  const safeValue = Number.isFinite(Number(value)) ? Number(value) : 0;
  const rounded = safeValue.toFixed(maximumFractionDigits).replace(/\.0$/, '');
  return `${rounded}%`;
}

function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.94) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Gambar belum bisa diproses.'));
    }, type, quality);
  });
}

function decodeImageFile(file) {
  if (typeof window !== 'undefined' && 'createImageBitmap' in window) {
    return createImageBitmap(file, { imageOrientation: 'from-image' }).catch(() => null);
  }
  return Promise.resolve(null);
}

function decodeImageWithElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Gambar tidak bisa dibaca di browser.'));
    };
    image.src = url;
  });
}

async function normalizeImageFile(file) {
  if (!file?.type?.startsWith('image/')) return file;

  const maxSide = 1600;
  const decoded = (await decodeImageFile(file)) || (await decodeImageWithElement(file));
  const width = decoded.width || 1;
  const height = decoded.height || 1;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));

  const context = canvas.getContext('2d');
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.fillStyle = '#f8eed3';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(decoded, 0, 0, canvas.width, canvas.height);
  if ('close' in decoded) decoded.close();

  const normalizedBlob = await canvasToBlob(canvas);
  const baseName = file.name?.replace(/\.[^.]+$/, '') || `chickpoo-image-${Date.now()}`;
  return new File([normalizedBlob], `${baseName}-normalized.jpg`, {
    type: 'image/jpeg',
    lastModified: file.lastModified || Date.now(),
  });
}

function formatWholePercent(value) {
  return `${Math.round(Number(value || 0))}%`;
}

function objectValidationPercent(result) {
  return result?.object_validation?.percentage_chicken_feces ?? 0;
}

function isLikelyFeces(result) {
  return result?.object_validation?.status === 'accepted';
}

function photoSuitabilityPercent(result) {
  return objectValidationPercent(result);
}

function resultTone(result) {
  if (!result) return 'idle';
  if (result.status === 'invalid_input') return 'invalid';
  if (result.predicted_class === 'healthy') return 'good';
  if (result.predicted_class === 'ncd') return 'danger';
  if (result.predicted_class === 'salmo') return 'orange';
  return 'amber';
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [activeSection, setActiveSection] = useState('home');
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const themeSwitchTimerRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      window.localStorage.setItem('theme', theme);
    } catch {
      // Theme still applies for the active session when storage is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    return () => {
      stopCamera();
      window.clearTimeout(themeSwitchTimerRef.current);
      document.documentElement.classList.remove('theme-changing');
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
      setCameraError('Kamera belum bisa menampilkan preview. Tutup lalu buka lagi.');
    });
  }, [cameraActive]);

  useEffect(() => {
    const sectionIds = ['home', 'scanner', 'workflow', 'model'];
    const updateActiveSection = () => {
      const current = sectionIds
        .map((id) => {
          const element = document.getElementById(id);
          if (!element) return { id, top: Number.POSITIVE_INFINITY };
          return { id, top: Math.abs(element.getBoundingClientRect().top - 120) };
        })
        .sort((a, b) => a.top - b.top)[0];
      if (current?.id) setActiveSection(current.id);
    };

    updateActiveSection();
    window.addEventListener('scroll', updateActiveSection, { passive: true });
    return () => window.removeEventListener('scroll', updateActiveSection);
  }, []);

  function scrollToSection(sectionId) {
    setActiveSection(sectionId);
    setMenuOpen(false);
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function updateSelectedFile(file) {
    if (!file) return;
    setError('');
    setResult(null);
    const normalizedFile = await normalizeImageFile(file);
    setSelectedFile(normalizedFile);
    setPreviewUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return URL.createObjectURL(normalizedFile);
    });
  }

  function handleFileInput(event) {
    void updateSelectedFile(event.target.files?.[0]).catch(() => {
      setError('Gambar belum bisa diproses. Coba gunakan JPG atau PNG yang lebih umum.');
    });
    event.target.value = '';
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file?.type?.startsWith('image/')) {
      void updateSelectedFile(file).catch(() => {
        setError('Gambar belum bisa diproses. Coba gunakan JPG atau PNG yang lebih umum.');
      });
    } else {
      setError('Gunakan file gambar seperti JPG, PNG, atau WEBP.');
    }
  }

  function handleDrag(event, active) {
    event.preventDefault();
    setDragActive(active);
  }

  async function useSampleImage(item) {
    const response = await fetch(item.src);
    const blob = await response.blob();
    await updateSelectedFile(new File([blob], item.filename, { type: blob.type || 'image/jpeg' }));
    scrollToSection('scanner');
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
      void updateSelectedFile(new File([blob], `chickpoo-camera-${Date.now()}.jpg`, { type: 'image/jpeg' }));
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
        throw new Error(payload.detail || 'Analisis belum bisa diproses.');
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || 'Layanan belum siap. Coba beberapa saat lagi.');
    } finally {
      setIsLoading(false);
    }
  }

  function resetInput() {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setCameraError('');
    setPreviewUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return '';
    });
  }

  function toggleTheme() {
    document.documentElement.classList.add('theme-changing');
    window.clearTimeout(themeSwitchTimerRef.current);
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
    themeSwitchTimerRef.current = window.setTimeout(() => {
      document.documentElement.classList.remove('theme-changing');
    }, 260);
  }

  return (
    <main className="app-shell">
      <AmbientBackground />
      <TopNav
        activeSection={activeSection}
        menuOpen={menuOpen}
        onNavigate={scrollToSection}
        theme={theme}
        onToggleTheme={toggleTheme}
        onToggleMenu={() => setMenuOpen((open) => !open)}
      />
      <HeroSection onStart={() => scrollToSection('scanner')} onLearn={() => scrollToSection('workflow')} />
      <ScannerSection
        selectedFile={selectedFile}
        previewUrl={previewUrl}
        cameraActive={cameraActive}
        cameraError={cameraError}
        dragActive={dragActive}
        isLoading={isLoading}
        result={result}
        error={error}
        fileInputRef={fileInputRef}
        videoRef={videoRef}
        handleFileInput={handleFileInput}
        handleDrop={handleDrop}
        handleDrag={handleDrag}
        startCamera={startCamera}
        stopCamera={stopCamera}
        capturePhoto={capturePhoto}
        submitPrediction={submitPrediction}
        resetInput={resetInput}
        useSampleImage={useSampleImage}
      />
      <WorkflowSection onStart={() => scrollToSection('scanner')} />
      <ImpactSection />
      <ModelSection onStart={() => scrollToSection('scanner')} />
      <AboutSection onStart={() => scrollToSection('scanner')} />
    </main>
  );
}

function AmbientBackground() {
  return <div className="ambient" aria-hidden="true" />;
}

function TopNav({ activeSection, menuOpen, onNavigate, theme, onToggleTheme, onToggleMenu }) {
  return (
    <header className={`topbar ${menuOpen ? 'menu-open' : ''}`}>
      <button className="brand" type="button" onClick={() => onNavigate('home')} title="Kembali ke beranda">
        <span className="brand-mark">
          <img src="/assets/poop.png" alt="" />
        </span>
        <span>
          <strong>ChickPoo</strong>
          <small>Smart Poultry Assistant</small>
        </span>
      </button>
      <button className="menu-toggle" type="button" onClick={onToggleMenu} aria-label="Buka menu navigasi" aria-expanded={menuOpen}>
        {menuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <nav className="nav-links" aria-label="Navigasi utama">
        {navItems.map((item) => (
          <button
            className={activeSection === item.id ? 'active' : ''}
            key={item.id}
            type="button"
            onClick={() => onNavigate(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <button
        className="theme-toggle"
        type="button"
        onClick={onToggleTheme}
        aria-label={`Ubah ke ${theme === 'dark' ? 'light mode' : 'dark mode'}`}
        title={`Ubah ke ${theme === 'dark' ? 'light mode' : 'dark mode'}`}
      >
        <span className="theme-toggle-icon" aria-hidden="true">
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </span>
        <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
      </button>
      <button className="nav-cta" type="button" onClick={() => onNavigate('scanner')}>
        <ScanSearch size={18} />
        <span>Mulai Scan</span>
      </button>
    </header>
  );
}

function HeroSection({ onStart, onLearn }) {
  return (
    <section className="hero-section" id="home">
      <div className="hero-copy">
        <div className="hero-badge">
          <Activity size={16} />
          <span>Your Smart Poultry Assistant</span>
        </div>
        <h1>Screening Kesehatan Ayam yang Lebih Cepat, Cerdas, dan Terarah.</h1>
        <p>
          ChickPoo membantu peternak memvalidasi foto, membaca kemungkinan kondisi kesehatan, dan menyajikan
          rekomendasi awal dalam satu alur yang sederhana.
        </p>
        <div className="hero-cta-zone">
          <div className="hero-actions">
            <button className="primary-action hero-primary" type="button" onClick={onStart}>
              <ScanSearch size={20} />
              <span>Mulai Scan</span>
              <ChevronRight className="cta-arrow" size={18} />
            </button>
            <button className="secondary-action hero-secondary" type="button" onClick={onLearn}>
              <span>Lihat Cara Kerja</span>
              <ArrowRight size={18} />
            </button>
          </div>
          <p className="hero-microcopy">
            <Sparkles size={15} />
            Screening awal dan rekomendasi tindakan berjalan dalam satu proses.
          </p>
        </div>
        <div className="hero-proof" aria-label="Ringkasan alur ChickPoo">
          <span>Alur sederhana</span>
          <span>Persentase transparan</span>
          <span>Rekomendasi tindakan</span>
        </div>
      </div>

      <aside className="hero-console-panel" aria-label="Mockup diagnostic console ChickPoo">
        <div className="diagnostic-console">
          <div className="console-topline">
            <span>
              <Activity size={16} />
              Live diagnostic console
            </span>
            <strong>Ready</strong>
          </div>

          <div className="console-visual">
            <img className="console-support-photo" src="/assets/ayam-putih.webp" alt="" aria-hidden="true" />
            <span className="console-grid" aria-hidden="true" />
            <span className="console-scan-core" aria-hidden="true" />
            <div className="console-status-card object">
              <ShieldCheck size={18} />
              <div>
                <span style={{ fontWeight: 760 }}>Risiko Terbaca</span>
                <strong style={{ fontWeight: 800, fontSize: "0.95rem" }}>Keputusan awal lebih cepat</strong>
              </div>
            </div>
            <div className="console-status-card disease">
              <BarChart3 size={18} />
              <div>
                <span>Disease Classifier Active</span>
                <strong>4 kelas</strong>
              </div>
            </div>
          </div>

          <div className="console-bottom">
            <div className="console-confidence">
              <div className="hero-score-ring" style={{ '--score': '352deg' }}>
                <span>{formatWholePercent(99)}</span>
              </div>
              <div>
                <p className="eyebrow">Confidence preview</p>
                <h3>Screening awal siap</h3>
                <p>Validasi, prediksi, dan rekomendasi diringkas dalam satu alur.</p>
              </div>
            </div>
            <div className="console-bars" aria-hidden="true">
              <span style={{ '--bar': '92%' }} />
              <span style={{ '--bar': '76%' }} />
              <span style={{ '--bar': '58%' }} />
            </div>
          </div>
        </div>
      </aside>
    </section>
  );
}

function ScannerSection({
  selectedFile,
  previewUrl,
  cameraActive,
  cameraError,
  dragActive,
  isLoading,
  result,
  error,
  fileInputRef,
  videoRef,
  handleFileInput,
  handleDrop,
  handleDrag,
  startCamera,
  stopCamera,
  capturePhoto,
  submitPrediction,
  resetInput,
  useSampleImage,
}) {
  return (
    <section className="scanner-section" id="scanner">
      <SectionIntro
        kicker="Scanner Diagnostik"
        title="Unggah foto feses ayam untuk memulai screening awal."
        text="Upload, preview, validasi objek, prediksi kondisi, dan rekomendasi awal berjalan dalam satu modul diagnostik."
      />

      <div className="scanner-grid">
        <div className="scanner-card glass-panel">
          <div className="device-header">
            <span>
              <ScanSearch size={18} />
              ChickPoo scanner
            </span>
            <strong>{isLoading ? 'Scanning' : selectedFile ? 'Ready' : 'Standby'}</strong>
          </div>
          <StepIndicator result={result} isLoading={isLoading} selectedFile={selectedFile} />
          <div
            className={`drop-zone ${dragActive ? 'dragging' : ''} ${previewUrl || cameraActive ? 'has-preview' : ''} ${isLoading ? 'is-scanning' : ''}`}
            onDragOver={(event) => handleDrag(event, true)}
            onDragLeave={(event) => handleDrag(event, false)}
            onDrop={handleDrop}
          >
            {cameraActive ? (
              <video ref={videoRef} className="preview-media camera-preview" muted playsInline />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Preview foto feses ayam" className="preview-media uploaded-preview" />
            ) : (
              <div className="drop-empty">
                <span className="drop-icon">
                  <ImagePlus size={42} />
                </span>
                <h3>Tarik gambar ke sini</h3>
                <p>Atau pilih foto dari perangkat. Gunakan gambar yang dekat, terang, dan fokus.</p>
              </div>
            )}
            <div className="scanner-reticle" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
            </div>
          </div>

          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} hidden />

          <div className="scanner-actions">
            <button className="secondary-action" type="button" onClick={() => fileInputRef.current?.click()}>
              <Upload size={18} />
              <span>Upload Gambar</span>
            </button>
            {cameraActive ? (
              <>
                <button className="secondary-action accent" type="button" onClick={capturePhoto}>
                  <Camera size={18} />
                  <span>Ambil Foto</span>
                </button>
                <button className="icon-action" type="button" onClick={stopCamera} title="Tutup kamera">
                  <X size={18} />
                </button>
              </>
            ) : (
              <button className="secondary-action" type="button" onClick={startCamera}>
                <Camera size={18} />
                <span>Kamera</span>
              </button>
            )}
          </div>

          <button className="scan-action" type="button" onClick={submitPrediction} disabled={isLoading || !selectedFile}>
            {isLoading ? <LoaderCircle className="spin" size={20} /> : <ScanSearch size={20} />}
            <span>{isLoading ? 'Foto sedang diperiksa...' : 'Analisis Sekarang'}</span>
          </button>

          {(selectedFile || result) && (
            <button className="reset-link" type="button" onClick={resetInput}>
              <RotateCcw size={16} />
              <span>Upload Ulang</span>
            </button>
          )}

          {cameraError && <p className="surface-alert">{cameraError}</p>}
          {error && <p className="surface-alert danger">{error}</p>}
        </div>

        <ResultPanel result={result} isLoading={isLoading} resetInput={resetInput} />
      </div>

      <SampleStrip useSampleImage={useSampleImage} />
    </section>
  );
}

function StepIndicator({ result, isLoading, selectedFile }) {
  const currentStep = useMemo(() => {
    if (result?.status === 'invalid_input') return 2;
    if (result) return 4;
    if (isLoading) return 2;
    if (selectedFile) return 1;
    return 0;
  }, [result, isLoading, selectedFile]);

  const steps = [
    { number: '01', label: 'Upload foto' },
    { number: '02', label: 'Validasi objek' },
    { number: '03', label: 'Prediksi penyakit' },
    { number: '04', label: 'Rekomendasi' },
  ];

  return (
    <div className="stepper" aria-label="Tahapan screening ChickPoo">
      {steps.map((step, index) => (
        <div className={`step-pill ${currentStep >= index + 1 ? 'active' : ''}`} key={step.number}>
          <span>{step.number}</span>
          <strong>{step.label}</strong>
        </div>
      ))}
    </div>
  );
}

function ResultPanel({ result, isLoading, resetInput }) {
  const StatusIcon = result ? statusIcon[result.status] || AlertTriangle : Gauge;
  const tone = resultTone(result);

  if (isLoading) {
    return (
      <aside className="result-card glass-panel loading-state">
        <div className="loading-orbit diagnostic-loader">
          <LoaderCircle className="spin" size={42} />
        </div>
        <h2>Foto sedang diperiksa...</h2>
        <p>Foto divalidasi terlebih dahulu agar objek yang salah tidak langsung diprediksi sebagai penyakit.</p>
      </aside>
    );
  }

  if (!result) {
    return (
      <aside className="result-card glass-panel empty-state">
        <div className="empty-orb">
          <Gauge size={42} />
        </div>
        <p className="eyebrow">Belum Ada Hasil</p>
        <h2>Unggah foto feses ayam untuk memulai screening awal.</h2>
        <p>Hasil validasi objek, prediksi penyakit, dan rekomendasi akan tampil di panel ini.</p>
      </aside>
    );
  }

  const invalidInput = result.status === 'invalid_input';
  const confidence = invalidInput ? photoSuitabilityPercent(result) : result.confidence_percentage;

  return (
    <aside className={`result-card glass-panel ${tone}`}>
      <div className="result-status">
        <span>
          <StatusIcon size={19} />
          {invalidInput ? 'Objek tidak sesuai' : 'Feses ayam terdeteksi'}
        </span>
        <strong>{result.recommendation.confidence_level}</strong>
      </div>

      <div className="result-flow">
        <span className="active">Validasi objek</span>
        <span className={!invalidInput ? 'active' : ''}>Prediksi penyakit</span>
        <span className={!invalidInput ? 'active' : ''}>Rekomendasi awal</span>
      </div>

      <div className="result-hero">
        <div className="score-ring" style={{ '--score': `${Math.min(Number(confidence || 0), 100) * 3.6}deg` }}>
          <span>{formatPercent(confidence)}</span>
        </div>
        <div>
          <p className="eyebrow">{invalidInput ? 'Status Foto' : 'Hasil Screening'}</p>
          <h2>{invalidInput ? 'Objek tidak sesuai' : result.predicted_label}</h2>
          <p>{result.status_message}</p>
        </div>
      </div>

      {invalidInput ? (
        <InvalidInputBlock result={result} resetInput={resetInput} />
      ) : (
        <DiseaseResult result={result} />
      )}

      <RecommendationCard recommendation={result.recommendation} />
    </aside>
  );
}

function InvalidInputBlock({ result, resetInput }) {
  return (
    <div className="invalid-block">
      <p>
        Gambar belum terdeteksi sebagai feses ayam. Silakan unggah foto yang lebih jelas.
      </p>
      <div className="mini-meter">
        <span>Skor kesesuaian foto</span>
        <strong>{formatPercent(photoSuitabilityPercent(result))}</strong>
      </div>
      <div className="progress-track">
        <span style={{ width: `${Math.max(photoSuitabilityPercent(result), 1)}%` }} />
      </div>
      <button className="secondary-action accent" type="button" onClick={resetInput}>
        <RotateCcw size={17} />
        <span>Upload Ulang</span>
      </button>
    </div>
  );
}

function DiseaseResult({ result }) {
  const lowConfidence = Number(result.confidence_percentage || 0) < 60;
  return (
    <div className="disease-panel">
      {lowConfidence && (
        <div className="soft-warning">
          Hasil belum cukup yakin. Coba unggah foto dengan pencahayaan lebih baik.
        </div>
      )}
      {isLikelyFeces(result) && (
        <div className="soft-warning">
          Foto terindikasi sesuai, tetapi hasil tetap sebaiknya dibaca sebagai indikasi awal.
        </div>
      )}
      <div className="probability-list">
        {result.probabilities.map((item) => (
          <div className={`probability-row ${item.key}`} key={item.key}>
            <div className="probability-meta">
              <span>{item.label}</span>
              <strong>{formatPercent(item.percentage)}</strong>
            </div>
            <div className="progress-track" aria-hidden="true">
              <span style={{ width: `${Math.max(item.percentage, 1)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecommendationCard({ recommendation }) {
  return (
    <article className="recommendation-card">
      <p className="eyebrow">Rekomendasi Awal</p>
      <h3>{recommendation.headline}</h3>
      <p>{recommendation.confidence_message}</p>
      <ul>
        {recommendation.actions.map((action) => (
          <li key={action}>
            <CheckCircle2 size={17} />
            <span>{action}</span>
          </li>
        ))}
      </ul>
      <small>Hasil ini adalah screening awal, bukan diagnosis final.</small>
    </article>
  );
}

function SampleStrip({ useSampleImage }) {
  return (
    <div className="sample-strip">
      <div>
        <p className="eyebrow">Contoh Cepat</p>
        <h3>Coba contoh cepat</h3>
      </div>
      <div className="sample-actions">
        {sampleImages.map((item) => (
          <button type="button" key={item.label} onClick={() => useSampleImage(item)} title={`Pakai contoh ${item.label}`}>
            <img src={item.src} alt={`Contoh ${item.label}`} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function WorkflowSection({ onStart }) {
  const steps = [
    { number: '01', title: 'Upload Foto', text: 'Masukkan foto feses ayam yang jelas.', Icon: Upload },
    { number: '02', title: 'Validasi Objek', text: 'Foto dicek apakah sudah sesuai.', Icon: ShieldCheck },
    { number: '03', title: 'Prediksi Kondisi', text: 'Jika valid, sistem membaca kemungkinan kondisi kesehatan.', Icon: BarChart3 },
    { number: '04', title: 'Rekomendasi Awal', text: 'Langkah awal ditampilkan secara ringkas dan mudah dipahami.', Icon: CheckCircle2 },
  ];

  return (
    <section className="workflow-section" id="workflow">
      <SectionIntro
        kicker="Cara Kerja ChickPoo"
        title="Dari Foto ke Rekomendasi dalam Satu Alur."
        text="ChickPoo menyederhanakan proses screening agar peternak dapat bergerak lebih cepat."
      />
      <div className="workflow-timeline">
        {steps.map(({ number, title, text, Icon }) => (
          <article className="timeline-node" key={number}>
            <span className="timeline-icon">
              <Icon size={20} />
            </span>
            <small>{number}</small>
            <h3>{title}</h3>
            <p>{text}</p>
          </article>
        ))}
      </div>
      <div className="why-card glass-panel">
        <Wheat size={30} />
        <div>
          <p className="eyebrow">Kenapa Penting?</p>
          <h3>Deteksi awal membantu peternak bergerak lebih cepat.</h3>
          <p>
            Penyakit yang terlambat dikenali dapat menyebar di kandang dan menambah kerugian.
            Screening awal memberi sinyal agar tindakan pencegahan bisa dimulai lebih cepat.
          </p>
        </div>
        <figure className="workflow-photo" aria-hidden="true">
          <img src="/assets/ayam-putih.webp" alt="" />
        </figure>
        <button className="primary-action" type="button" onClick={onStart}>
          <span>Mulai Scan</span>
          <ChevronRight size={18} />
        </button>
      </div>
    </section>
  );
}

function ImpactSection() {
  return (
    <section className="impact-section" aria-label="Konteks peternak ChickPoo">
      <div className="impact-card glass-panel">
        <figure className="impact-photo">
          <img src="/assets/peternak-ayam.webp" alt="Peternak memeriksa ayam di kandang" />
        </figure>
        <div className="impact-copy">
          <p className="eyebrow">Human Trust</p>
          <h2>Dibuat untuk membantu peternak bergerak lebih cepat.</h2>
          <p>
            ChickPoo membantu peternak melakukan screening awal dari foto feses ayam, sehingga
            tindakan pencegahan dapat dilakukan lebih cepat sebelum kondisi memburuk.
          </p>
          <div className="impact-points" aria-label="Nilai utama ChickPoo">
            <span>Screening awal</span>
            <span>Alur sederhana</span>
            <span>Rekomendasi siap dibaca</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModelSection({ onStart }) {
  return (
    <section className="model-section" id="model">
      <SectionIntro
        kicker="Product Analytics"
        title="Screening dua tahap untuk hasil yang lebih terkontrol."
        text="Model pertama memvalidasi objek, lalu model kedua membaca kemungkinan kondisi kesehatan jika foto dinyatakan sesuai."
      />

      <div className="model-console glass-panel">
        <div className="model-console-head">
          <div>
            <p className="eyebrow">Performance Console</p>
            <h3>Data performa disajikan sebagai panel produk, bukan laporan tempelan.</h3>
          </div>
          <button className="primary-action" type="button" onClick={onStart}>
            <ScanSearch size={18} />
            <span>Mulai Scan</span>
          </button>
        </div>

        <div className="metrics-grid">
          {modelMetrics.map((item) => (
            <article className="metric-card" key={item.label}>
              <span className="metric-icon">
                <item.Icon size={18} />
              </span>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <p>{item.note}</p>
            </article>
          ))}
        </div>

        <div className="model-layout">
          <section className="model-explain">
            <p className="eyebrow">Model Pipeline</p>
            <h3>Mengapa dua tahap?</h3>
            <p className="model-note">
              Validasi objek menahan gambar yang tidak sesuai. Klasifikasi penyakit hanya berjalan
              jika objek valid. Pendekatan ini membantu mengurangi prediksi yang tidak masuk akal.
            </p>
          <div className="pipeline-steps">
            <article>
              <span>01</span>
              <strong>Validasi objek</strong>
              <p>Memastikan gambar yang masuk benar-benar menyerupai feses ayam.</p>
            </article>
            <article>
              <span>02</span>
              <strong>Klasifikasi penyakit</strong>
              <p>Jika valid, model membaca kemungkinan healthy, coccidiosis, salmonella, atau Newcastle Disease.</p>
            </article>
          </div>
          <div className="class-list">
            {classInfo.map((item) => (
              <article className={`class-item ${item.tone}`} key={item.key}>
                <strong>{item.label}</strong>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
          </section>

          <div className="matrix-stack">
            {confusionMatrices.map((matrix) => (
              <ConfusionMatrixHeatmap
                key={matrix.title}
                title={matrix.title}
                classes={matrix.labels}
                matrix={matrix.data}
                caption="Nilai diagonal menunjukkan prediksi yang benar, sedangkan nilai di luar diagonal menunjukkan kesalahan klasifikasi."
              />
            ))}
          </div>
        </div>

        <article className="insight-card">
          <p className="eyebrow">Insight</p>
          <h3>Kenapa dua tahap?</h3>
          <p>
            Model digunakan sebagai screening awal dan tetap membutuhkan validasi petugas kesehatan
            hewan untuk keputusan pengobatan.
          </p>
        </article>
      </div>
    </section>
  );
}

function ConfusionMatrixHeatmap({ title, classes, matrix, caption }) {
  const maxValue = Math.max(...matrix.flat());

  return (
    <article className={`matrix-card matrix-${classes.length}`}>
      <div className="matrix-card-head">
        <div>
          <p className="eyebrow">{title}</p>
          <h3>Heatmap evaluasi</h3>
        </div>
        <span>Baris = True Label, Kolom = Predicted Label</span>
      </div>
      <div
        className="matrix-grid"
        style={{ '--matrix-count': classes.length }}
        aria-label={title}
      >
        <span className="matrix-axis">
          <em>True</em>
          <strong>Predicted</strong>
        </span>
        {classes.map((label) => (
          <span className="matrix-label predicted" key={`pred-${label}`}>
            {formatMatrixLabel(label)}
          </span>
        ))}
        {classes.map((rowLabel, rowIndex) => (
          <Fragment key={rowLabel}>
            <span className="matrix-label true">{formatMatrixLabel(rowLabel)}</span>
            {matrix[rowIndex].map((value, columnIndex) => {
              const intensity = maxValue ? value / maxValue : 0;
              const diagonal = rowIndex === columnIndex;
              return (
                <span
                  className={`matrix-cell ${diagonal ? 'diagonal' : ''}`}
                  key={`${rowLabel}-${classes[columnIndex]}`}
                  style={{
                    '--intensity': intensity,
                    '--heat-alpha': 0.08 + intensity * 0.34,
                    '--cream-alpha': 0.12 + intensity * 0.63,
                  }}
                  title={`True: ${formatMatrixLabel(rowLabel)}, Predicted: ${formatMatrixLabel(classes[columnIndex])}, Count: ${value}`}
                >
                  {value}
                </span>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="matrix-caption">{caption}</p>
    </article>
  );
}

function formatMatrixLabel(label) {
  const labels = {
    not_chicken_feces: 'Bukan feses',
    chicken_feces: 'Feses ayam',
    cocci: 'Cocci',
    healthy: 'Sehat',
    ncd: 'NCD',
    salmo: 'Salmo',
  };
  return labels[label] || label;
}

function AboutSection({ onStart }) {
  return (
    <section className="about-section">
      <div className="closing-card glass-panel">
        <div className="closing-copy">
          <p className="eyebrow">Tentang ChickPoo</p>
          <h2>Screening awal yang praktis, hangat, dan tetap bertanggung jawab.</h2>
          <p>
            ChickPoo membantu membaca tanda awal dari foto feses, menyajikan tingkat keyakinan,
            dan memberi rekomendasi awal yang mudah dipahami. Untuk keputusan pengobatan, tetap
            konsultasikan dengan petugas kesehatan hewan.
          </p>
          <button className="primary-action" type="button" onClick={onStart}>
            <span>Mulai Scan</span>
            <ArrowRight size={18} />
          </button>
        </div>
        <div className="closing-diagnostics" aria-label="Ringkasan nilai ChickPoo">
          <article>
            <span>01</span>
            <strong>Validasi foto</strong>
            <p>Gambar dicek lebih dulu sebelum kondisi dibaca.</p>
          </article>
          <article>
            <span>02</span>
            <strong>Prediksi awal</strong>
            <p>Hasil ditampilkan dengan tingkat keyakinan yang transparan.</p>
          </article>
          <article>
            <span>03</span>
            <strong>Rekomendasi</strong>
            <p>Arahan awal dibuat ringkas untuk membantu tindakan berikutnya.</p>
          </article>
        </div>
      </div>
    </section>
  );
}

function SectionIntro({ kicker, title, text }) {
  return (
    <div className="section-intro">
      <p className="eyebrow">{kicker}</p>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}

export default App;
