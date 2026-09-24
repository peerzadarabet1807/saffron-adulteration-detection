import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ProbabilityBars } from '../../src/components/ProbabilityBars';
import { UploadCard } from '../../src/components/UploadCard';
import { LowConfidenceWarning, VerdictBanner } from '../../src/components/Verdict';
import { interpret } from '../../src/lib/prediction';

describe('ProbabilityBars', () => {
  it('renders five labelled bars, most likely first', () => {
    const { ranked } = interpret([0.029, 0.016, 0.05, 0.096, 0.809]);
    render(<ProbabilityBars ranked={ranked} />);
    const rows = screen.getAllByTestId('prob-row');
    expect(rows).toHaveLength(5);
    expect(rows.map((r) => r.dataset.class)).toEqual(['C5', 'C4', 'C3', 'C1', 'C2']);
    expect(within(rows[0]).getByText('80.9%')).toBeInTheDocument();
    // group identity is written out, not only colour-coded
    expect(within(rows[0]).getByText('· Counterfeit')).toBeInTheDocument();
    expect(within(rows[3]).getByText('· Pure')).toBeInTheDocument();
  });
});

describe('verdict and warnings', () => {
  it('shows "Likely counterfeit" from the group probabilities', () => {
    render(<VerdictBanner interpretation={interpret([0.029, 0.016, 0.05, 0.096, 0.809])} />);
    expect(screen.getByText('Likely counterfeit')).toBeInTheDocument();
    expect(screen.getByText(/95\.5% of the probability/)).toBeInTheDocument();
  });

  it('warns on low confidence and reminds that the model always picks a class', () => {
    render(<LowConfidenceWarning interpretation={interpret([0.033, 0.551, 0.026, 0.006, 0.384])} />);
    const note = screen.getByTestId('low-confidence');
    expect(note).toHaveTextContent('only 55.1% probability');
    expect(note).toHaveTextContent('always picks one of its five classes');
  });

  it('stays silent for a confident prediction', () => {
    const { container } = render(
      <LowConfidenceWarning interpretation={interpret([0.93, 0.03, 0.02, 0.01, 0.01])} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

describe('UploadCard', () => {
  it('passes a chosen file to the classifier and is keyboard reachable', async () => {
    const onFile = vi.fn();
    render(<UploadCard busy={false} onFile={onFile} status={null} />);
    const button = screen.getByRole('button', { name: /choose a photo/i });
    await userEvent.tab();
    expect(button).toHaveFocus();
    const photo = new File(['x'], 'thread.jpg', { type: 'image/jpeg' });
    await userEvent.upload(screen.getByTestId('file-input'), photo);
    expect(onFile).toHaveBeenCalledWith(photo, 'upload');
  });

  it('announces validation errors', () => {
    render(
      <UploadCard
        busy={false}
        onFile={() => {}}
        status={null}
        error="“notes.txt” doesn't look like a photo."
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent("doesn't look like a photo");
  });

  it('states the privacy guarantee', () => {
    render(<UploadCard busy={false} onFile={() => {}} status={null} />);
    expect(screen.getByText(/never leaves your device/)).toBeInTheDocument();
  });
});
