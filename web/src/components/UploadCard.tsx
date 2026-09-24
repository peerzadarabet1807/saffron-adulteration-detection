import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type ReactNode } from 'react';
import type { SourceKind } from '../hooks/useClassifier';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { MAX_FILE_BYTES, formatBytes } from '../lib/fileValidation';
import { CameraIcon, LockIcon, UploadIcon } from './icons';

const ACCEPT = 'image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif,.jpg,.jpeg,.png,.webp';

interface Props {
  busy: boolean;
  error?: string;
  onFile: (file: File, kind: SourceKind) => void;
  status: ReactNode;
}

export function UploadCard({ busy, error, onFile, status }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const coarse = useMediaQuery('(pointer: coarse)'); // phones and tablets get a camera button

  // Paste anywhere on the page (Ctrl/Cmd+V), unless the user is typing in a field.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('input, textarea, [contenteditable="true"]')) return;
      const file = Array.from(e.clipboardData?.files ?? [])[0];
      if (file) {
        e.preventDefault();
        onFile(file, 'paste');
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [onFile]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file, 'drop');
  };

  const onPick = (kind: SourceKind) => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow choosing the same file again
    if (file) onFile(file, kind);
  };

  return (
    <div className="card p-4 sm:p-5" id="upload">
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={onDrop}
        data-testid="dropzone"
        className={`relative flex flex-col items-center rounded-2xl border-2 border-dashed px-5 py-8 text-center transition-colors sm:py-10 ${
          dragging ? 'border-crimson bg-fake-soft' : 'border-line-strong bg-surface-2/60'
        }`}
      >
        <div className="mb-4 grid size-14 place-items-center rounded-full bg-surface text-2xl text-crimson-ink shadow-sm ring-1 ring-line">
          <UploadIcon />
        </div>
        <p className="font-serif text-xl font-semibold">Drop a photo of one saffron thread</p>
        <p className="mt-1 text-sm text-ink-3">
          JPEG, PNG, WebP or HEIC · up to {formatBytes(MAX_FILE_BYTES)}
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-3">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => fileInput.current?.click()}
            disabled={busy}
          >
            <UploadIcon /> Choose a photo
          </button>
          {coarse && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => cameraInput.current?.click()}
              disabled={busy}
            >
              <CameraIcon /> Take a photo
            </button>
          )}
        </div>
        <p className="mt-4 hidden text-xs text-ink-3 sm:block">
          or paste an image with <kbd className="rounded border border-line-strong px-1 font-sans">Ctrl</kbd>/
          <kbd className="rounded border border-line-strong px-1 font-sans">⌘</kbd>+
          <kbd className="rounded border border-line-strong px-1 font-sans">V</kbd>
        </p>
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          tabIndex={-1}
          aria-label="Choose a photo of a saffron thread"
          data-testid="file-input"
          onChange={onPick('upload')}
        />
        <input
          ref={cameraInput}
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          tabIndex={-1}
          aria-label="Take a photo of a saffron thread"
          onChange={onPick('camera')}
        />
      </div>

      {error && (
        <p
          role="alert"
          className="mt-4 rounded-xl border border-fake/40 bg-fake-soft px-4 py-3 text-sm text-fake-ink"
          data-testid="upload-error"
        >
          {error}
        </p>
      )}

      <div className="mt-4 flex items-start gap-2.5 text-sm text-ink-2">
        <LockIcon className="mt-0.5 shrink-0 text-base text-gold-ink" />
        <p>
          <strong className="font-semibold text-ink">Your photo never leaves your device;</strong> the model
          runs in your browser.
        </p>
      </div>
      <div className="mt-4 border-t border-line pt-4">{status}</div>
    </div>
  );
}
