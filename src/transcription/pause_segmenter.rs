//! Pause detection over 16 kHz meeting audio, so rolling transcription cuts
//! chunks at natural pauses.
//!
//! Speech is any 20 ms frame at least 12 dB above an adaptive noise floor and
//! never quieter than -55 dBFS. The floor starts at the first frame, falls
//! immediately, and rises slowly, so speech between word gaps cannot drag it
//! up to the speech level. A piece ends in the middle of a pause, leaving both
//! neighbours some silence as padding.

use std::time::Duration;

const SAMPLE_RATE: usize = 16_000;
pub(crate) const FRAME_SAMPLES: usize = SAMPLE_RATE / 50;
const SPEECH_MARGIN_DB: f32 = 12.0;
/// How far above the floor a frame may sit and still count as quiet.
const QUIET_MARGIN_DB: f32 = 4.0;
const MINIMUM_SPEECH_DB: f32 = -55.0;
/// Below this a frame is quiet whatever the floor. Lower than
/// `MINIMUM_SPEECH_DB`: a quiet built-in microphone can carry whole
/// dictations under -55 dBFS (#841).
const QUIET_MINIMUM_DB: f32 = -70.0;
const LOWEST_FLOOR_DB: f32 = -90.0;
/// About 3 dB per second at 50 frames per second.
const FLOOR_RISE_DB_PER_FRAME: f32 = 0.06;
const MINIMUM_PIECE: Duration = Duration::from_secs(3);
const PAUSE: Duration = Duration::from_millis(600);
const LONG_PIECE: Duration = Duration::from_secs(8);
const LONG_PIECE_PAUSE: Duration = Duration::from_millis(300);

/// Classifies consecutive 20 ms frames of 16 kHz audio as speech or not
/// against the adaptive noise floor described above.
#[derive(Default)]
pub(crate) struct SpeechFrames {
    floor_db: Option<f32>,
}

impl SpeechFrames {
    /// Whether `frame`, the next `FRAME_SAMPLES` samples, is speech.
    pub(crate) fn is_speech(&mut self, frame: &[f32]) -> bool {
        let (level_db, floor_db) = self.measure(frame);
        level_db > (floor_db + SPEECH_MARGIN_DB).max(MINIMUM_SPEECH_DB)
    }

    /// Whether `frame` sits at the noise floor (within `QUIET_MARGIN_DB`) or
    /// below `QUIET_MINIMUM_DB`. Stricter than "not speech": speech only a
    /// few dB above steady room noise, or from a quiet microphone, is not
    /// quiet, so audio that is dropped as quiet is never quiet speech the
    /// speech margin missed.
    pub(crate) fn is_quiet(&mut self, frame: &[f32]) -> bool {
        let (level_db, floor_db) = self.measure(frame);
        level_db <= (floor_db + QUIET_MARGIN_DB).max(QUIET_MINIMUM_DB)
    }

    /// The frame's DC-removed level and the noise floor after it, both in
    /// dBFS. Some microphones (a Ryzen built-in digital microphone, #841)
    /// carry a DC offset far louder than the speech on it; counted as level,
    /// it pins every frame to the floor and the whole dictation reads as a
    /// pause. The dictation silence gate removes DC the same way.
    fn measure(&mut self, frame: &[f32]) -> (f32, f32) {
        let count = frame.len() as f32;
        let mean = frame.iter().sum::<f32>() / count;
        let energy = frame
            .iter()
            .map(|sample| (sample - mean) * (sample - mean))
            .sum::<f32>()
            / count;
        let level_db = if energy.is_finite() {
            10.0 * energy.max(1e-18).log10()
        } else {
            0.0
        };
        let floor_db = match self.floor_db {
            Some(floor_db) if level_db >= floor_db => floor_db + FLOOR_RISE_DB_PER_FRAME,
            _ => level_db.max(LOWEST_FLOOR_DB),
        };
        self.floor_db = Some(floor_db);
        (level_db, floor_db)
    }
}

#[derive(Default)]
pub struct PauseSegmenter {
    speech: SpeechFrames,
    frame: Vec<f32>,
    position: usize,
    piece_start: usize,
    silent_frames: usize,
    heard_speech: bool,
}

impl PauseSegmenter {
    /// Consumes contiguous samples and returns the last cut found within them,
    /// as an absolute sample offset from the first sample ever pushed.
    pub fn push(&mut self, mut samples: &[f32]) -> Option<usize> {
        let mut cut = None;
        while !samples.is_empty() {
            let take = samples.len().min(FRAME_SAMPLES - self.frame.len());
            self.frame.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.frame.len() == FRAME_SAMPLES {
                cut = self.complete_frame().or(cut);
                self.frame.clear();
            }
        }
        cut
    }

    fn complete_frame(&mut self) -> Option<usize> {
        self.position += FRAME_SAMPLES;
        if self.speech.is_speech(&self.frame) {
            self.silent_frames = 0;
            self.heard_speech = true;
            return None;
        }
        self.silent_frames += 1;
        let piece = self.position - self.piece_start;
        let pause = if piece >= samples(LONG_PIECE) {
            LONG_PIECE_PAUSE
        } else {
            PAUSE
        };
        if !self.heard_speech
            || piece < samples(MINIMUM_PIECE)
            || self.silent_frames * FRAME_SAMPLES < samples(pause)
        {
            return None;
        }
        let cut = self.position - self.silent_frames * FRAME_SAMPLES / 2;
        self.piece_start = cut;
        self.silent_frames = 0;
        self.heard_speech = false;
        Some(cut)
    }
}

const fn samples(duration: Duration) -> usize {
    duration.as_millis() as usize * SAMPLE_RATE / 1_000
}

/// Pause audio Labs › Skip pauses keeps at the start of each long pause, so
/// the words before it still end on silence.
pub(crate) const CLIP_PAUSE_HEAD_SAMPLES: usize = SAMPLE_RATE * 2 / 5;
/// Pause audio kept just before speech resumes.
pub(crate) const CLIP_PAUSE_TAIL_SAMPLES: usize = SAMPLE_RATE * 3 / 10;

/// Labs › Skip pauses: shortens every pause in a whole 16 kHz clip longer
/// than head plus tail (700 ms) to its first 400 ms and last 300 ms, so the
/// model reads less audio, and returns how many samples it removed. Only
/// frames at the noise floor count as pause (`SpeechFrames::is_quiet`), the
/// test `PauseCap` uses on live streams. A clip with no frame above the floor
/// is left whole: the floor misjudged the microphone (#841), not a silent
/// dictation, which the silence gate already discarded.
pub(crate) fn shorten_pauses(clip: &mut Vec<f32>) -> usize {
    let keep = CLIP_PAUSE_HEAD_SAMPLES + CLIP_PAUSE_TAIL_SAMPLES;
    let mut speech = SpeechFrames::default();
    let mut cuts = Vec::new();
    let mut pause_start = None;
    let mut heard = false;
    for (index, frame) in clip.as_chunks::<FRAME_SAMPLES>().0.iter().enumerate() {
        if speech.is_quiet(frame) {
            pause_start.get_or_insert(index * FRAME_SAMPLES);
            continue;
        }
        heard = true;
        let at = index * FRAME_SAMPLES;
        if let Some(start) = pause_start.take()
            && at - start > keep
        {
            cuts.push(start + CLIP_PAUSE_HEAD_SAMPLES..at - CLIP_PAUSE_TAIL_SAMPLES);
        }
    }
    if !heard {
        return 0;
    }
    // A pause still open at the end runs through the partial last frame.
    if let Some(start) = pause_start
        && clip.len() - start > keep
    {
        cuts.push(start + CLIP_PAUSE_HEAD_SAMPLES..clip.len() - CLIP_PAUSE_TAIL_SAMPLES);
    }
    // Compact in place: each kept run moves down over the cuts before it.
    let mut write = 0;
    let mut read = 0;
    for cut in &cuts {
        clip.copy_within(read..cut.start, write);
        write += cut.start - read;
        read = cut.end;
    }
    clip.copy_within(read.., write);
    write += clip.len() - read;
    let removed = clip.len() - write;
    clip.truncate(write);
    removed
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic noise at a target RMS level in dBFS.
    pub(crate) fn noise(duration_ms: usize, level_db: f32, seed: &mut u32) -> Vec<f32> {
        let amplitude = 10f32.powf(level_db / 20.0) * 3f32.sqrt();
        (0..duration_ms * SAMPLE_RATE / 1_000)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                amplitude * ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0)
            })
            .collect()
    }

    /// Alternating speech and pauses: `(speech_ms, pause_ms)`. Speech is
    /// syllable-like bursts separated by 60 ms dips to the room level.
    pub(crate) fn utterance(parts: &[(usize, usize)], floor_db: f32) -> Vec<f32> {
        let mut seed = 7;
        let mut audio = Vec::new();
        let room = |audio: &mut Vec<f32>, duration_ms: usize, seed: &mut u32| {
            if floor_db.is_finite() {
                audio.extend(noise(duration_ms, floor_db, seed));
            } else {
                audio.resize(audio.len() + duration_ms * SAMPLE_RATE / 1_000, 0.0);
            }
        };
        for &(speech, pause) in parts {
            let mut remaining = speech;
            while remaining > 0 {
                let burst = if remaining <= 360 { remaining } else { 240 };
                audio.extend(noise(burst, -20.0, &mut seed));
                remaining -= burst;
                if remaining > 0 {
                    room(&mut audio, 60, &mut seed);
                    remaining -= 60;
                }
            }
            room(&mut audio, pause, &mut seed);
        }
        audio
    }

    fn cuts(audio: &[f32], chunk: usize) -> Vec<usize> {
        let mut segmenter = PauseSegmenter::default();
        audio
            .chunks(chunk)
            .filter_map(|chunk| segmenter.push(chunk))
            .collect()
    }

    fn ms(samples: usize) -> usize {
        samples * 1_000 / SAMPLE_RATE
    }

    #[test]
    fn cuts_in_the_middle_of_a_pause_after_a_long_enough_piece() {
        let audio = utterance(&[(4_000, 1_000), (2_000, 0)], -70.0);
        let found = cuts(&audio, 147);
        assert_eq!(found.len(), 1);
        // Speech ends at 4.0 s; the 600 ms pause completes at 4.6 s.
        assert_eq!(ms(found[0]), 4_300);
    }

    #[test]
    fn short_pieces_wait_for_a_later_pause() {
        let audio = utterance(&[(800, 650), (800, 650), (2_000, 800), (500, 0)], -70.0);
        let found = cuts(&audio, 320);
        assert_eq!(found.len(), 1, "{found:?}");
        // The first two pauses close pieces under three seconds.
        assert_eq!(ms(found[0]), 5_200);
    }

    #[test]
    fn short_pauses_split_only_long_pieces() {
        let audio = utterance(&[(5_000, 400), (4_000, 400), (1_000, 0)], -70.0);
        let found = cuts(&audio, 480);
        assert_eq!(found.len(), 1, "{found:?}");
        // 400 ms is too short at 5.4 s but enough after 8 s.
        assert_eq!(ms(found[0]), 9_550);
    }

    #[test]
    fn noise_floor_adapts_to_room_noise() {
        let audio = utterance(&[(4_000, 1_000), (4_000, 1_000), (500, 0)], -40.0);
        assert_eq!(cuts(&audio, 320).len(), 2);

        // A noisy room with no speech never cuts.
        let mut seed = 3;
        assert!(cuts(&noise(20_000, -40.0, &mut seed), 320).is_empty());
    }

    #[test]
    fn digital_silence_counts_as_a_pause_and_speech_still_registers_after_it() {
        let audio = utterance(
            &[(4_000, 1_000), (4_000, 2_000), (4_000, 1_000), (500, 0)],
            f32::NEG_INFINITY,
        );
        let found = cuts(&audio, 512);
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(ms(found[0]), 4_300);
        assert_eq!(ms(found[1]), 9_300);
    }

    #[test]
    fn leading_silence_alone_never_cuts() {
        let mut audio = vec![0.0; SAMPLE_RATE * 5];
        audio.extend(utterance(&[(2_000, 0)], -70.0));
        assert!(cuts(&audio, 320).is_empty());
    }

    #[test]
    fn skip_pauses_keeps_the_ends_of_a_long_pause_and_every_word() {
        let audio = utterance(&[(2_000, 3_000), (2_000, 0)], -70.0);
        let mut clip = audio.clone();
        let removed = shorten_pauses(&mut clip);
        // 3 s of pause becomes its first 400 ms and last 300 ms.
        assert_eq!(ms(removed), 2_300);
        assert_eq!(clip.len(), audio.len() - removed);
        let before = samples(Duration::from_millis(2_400));
        assert_eq!(clip[..before], audio[..before]);
        let after = samples(Duration::from_millis(2_300));
        assert_eq!(clip[clip.len() - after..], audio[audio.len() - after..]);
    }

    #[test]
    fn skip_pauses_leaves_short_pauses_and_word_gaps_alone() {
        let audio = utterance(&[(2_000, 700), (2_000, 600), (2_000, 0)], -70.0);
        let mut clip = audio.clone();
        assert_eq!(shorten_pauses(&mut clip), 0);
        assert_eq!(clip, audio);
    }

    #[test]
    fn skip_pauses_shortens_leading_trailing_and_noisy_pauses() {
        let mut audio = utterance(&[(0, 2_000), (2_000, 2_000), (2_000, 2_000)], -40.0);
        let length = audio.len();
        assert_eq!(ms(shorten_pauses(&mut audio)), 3 * 1_300);
        assert_eq!(
            audio.len(),
            length - samples(Duration::from_millis(3 * 1_300))
        );
    }

    #[test]
    fn skip_pauses_leaves_a_clip_that_never_rises_above_the_floor_whole() {
        // A quiet microphone whose speech never clears the floor (#841).
        let mut seed = 5;
        let audio = noise(8_000, -75.0, &mut seed);
        let mut clip = audio.clone();
        assert_eq!(shorten_pauses(&mut clip), 0);
        assert_eq!(clip, audio);
    }
}
