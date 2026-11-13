[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000112-blue)](https://doi.org/10.82901/nemar.nm000112)

# FACED - Finer-grained Affective Computing EEG Dataset

## Introduction

The Finer-grained Affective Computing EEG Dataset (FACED) contains scalp EEG recordings from 123 healthy participants who watched 28 emotion-eliciting video clips designed to evoke nine different emotion categories. The dataset includes four negative emotions (anger, fear, disgust, sadness) from Ekman's basic emotions and four positive emotions (amusement, inspiration, joy, tenderness) selected based on recent psychological and neuroscience progress and application needs. Participants provided detailed self-reported emotion ratings on 12 dimensions: eight emotions, arousal, valence, liking, and familiarity. The dataset is designed to facilitate cross-subject affective computing research and development of EEG-based emotion recognition algorithms for real-world applications.

## Overview of the experiment

Participants (123 subjects, 75 female, ages 17-38, mean=23.2 years) were seated 60 cm from a 22-inch LCD monitor in a regular office environment. Each trial consisted of: (1) a 5-second fixation cross, (2) a video clip of varying length (typically 30-60 seconds), and (3) subjective emotional rating on 12 items (anger, fear, disgust, sadness, amusement, inspiration, joy, tenderness, valence, arousal, liking, familiarity) on a continuous 0-7 scale, followed by at least 30 seconds rest. Video clips were presented in blocks: three positive blocks, three negative blocks, and one neutral block, with 20 arithmetic problems between blocks to minimize carryover effects. The 28 video clips were designed to target nine emotion categories, with randomized presentation order across participants. EEG was recorded using a 32-channel biosignal recording system sampled at either 1000 Hz (92 subjects) or 250 Hz (31 subjects), with channels positioned according to the International 10-20 system. Signal units were recorded in either Volts or microVolts depending on the hardware configuration used.

**Video stimulus information:**
The dataset includes 28 video clips designed to elicit nine emotion categories (Trigger values 1–28):
- Anger (Videos 1-3): Durations 73-81 seconds, negative valence
- Disgust (Videos 4-6): Durations 69-91 seconds, negative valence
- Fear (Videos 7-9): Durations 56-106 seconds, negative valence
- Sadness (Videos 10-12): Durations 45-82 seconds, negative valence
- Neutral (Videos 13-16): Durations 35-43 seconds, neutral valence
- Amusement (Videos 17-19): Durations 56-73 seconds, positive valence
- Inspiration (Videos 20-22): Durations 76-129 seconds, positive valence
- Joy (Videos 23-25): Durations 34-68 seconds, positive valence
- Tenderness (Videos 26-28): Durations 54-77 seconds, positive valence

Metadata for each video (duration, source film, source database, valence, targeted emotion) is read from Stimuli_info.xlsx.

**Event markers (from evt.bdf annotations):**
- 100: Task/block start
- 101: Video onset
- 102: Video offset
- 1–28: Video index (appears just before 101, used to link to stimulus metadata)
- 201/202: Block boundary markers
- "Start Impedance" / "Stop Impedance": Technical markers (ignored)

The conversion script reads evt.bdf annotations for each subject, parses video presentation spans (from video index + 101 to 102), and creates MNE Annotations with the source film title (video_title) as description. These annotations are exported to BIDS events.tsv with extra columns:
- emotion_label: targeted emotion category (Anger, Disgust, Fear, Sadness, Neutral, Amusement, Inspiration, Joy, Tenderness)
- binary_label: positive/negative/neutral classification
- video_index: 1–28
- Self-reported ratings (Joy, Tenderness, Inspiration, Amusement, Anger, Disgust, Fear, Sadness, Arousal, Valence, Familiarity, Liking)

## Description of the preprocessing if any

Raw BDF files from the biosignal recording system have been converted to BIDS format. Channel names are standardized to match the International 10-20 nomenclature. Subjects have been assigned numeric IDs (sub-000 through sub-122) corresponding to their original subject designations in the dataset. Recording dates have been set to a default value (2023-01-01) due to privacy considerations, while time relationships between files are preserved. Subject demographic information (age, sex) has been extracted from the Recording_info.csv file and properly formatted for BIDS.

Stimulus timing information from the evt.bdf event files has been parsed and enriched with metadata from Stimuli_info.xlsx. Each video presentation is annotated with the targeted emotion category (Anger, Disgust, Fear, Sadness, Neutral, Amusement, Inspiration, Joy, Tenderness) and includes self-reported ratings from After_remarks.mat when available.

## Citation

When using this dataset, please cite:

1. Chen, J., Wang, X., Huang, C., Hu, X., Shen, X., & Zhang, D. (2023). A Large Finer-grained Affective Computing EEG Dataset. Scientific Data, 10, 740. https://doi.org/10.1038/s41597-023-02650-w

2. The data: FACED, Synapse project syn50614194, https://doi.org/10.7303/syn50614194 (DataCite creator TSINGHUA_EMOTION_BCI; the project was created on 2023-01-05 by the Synapse account THU-EMOTION-BCI). The paper's "Data Records" section names this project as the FACED repository, and the project's own wiki names the paper above as its data descriptor.

**Data curators:**
Pierre Guetschel (BIDS conversion)

Original data collection team:
- Jingjing Chen, Xiaobin Wang, Chen Huang, Xin Hu, Xinke Shen, Dan Zhang (Tsinghua University; authors of the data descriptor above, which credits the data collection, "Investigation", to Xiaobin Wang and Xin Hu)


---

## Automatic report

*Report automatically generated by `mne_bids.make_report()`.*

>  The FACED - Finer-grained Affective Computing EEG Dataset dataset was created
by Jingjing Chen, Xiaobin Wang, Chen Huang, Xin Hu, Xinke Shen, and Dan Zhang and conforms to BIDS version
1.7.0. This report was generated with MNE-BIDS
(https://doi.org/10.21105/joss.01896). The dataset consists of 123 participants
(comprised of 48 male and 75 female participants; handedness were all unknown;
ages ranged from 17.0 to 38.0 (mean = 22.94, std = 4.66)) . Data was recorded
using an EEG system (Biosemi) sampled at 1000.0, and 250.0 Hz with line noise at
n/a Hz. There were 123 scans in total. Recording durations ranged from 3468.0 to
6743.0 seconds (mean = 4544.83, std = 647.24), for a total of 559013.71 seconds
of data recorded over all scans. For each dataset, there were on average 32.0
(std = 0.0) recording channels per scan, out of which 32.0 (std = 0.0) were used
in analysis (0.0 +/- 0.0 were removed from analysis).

## Stimuli (added 2026-10-08, metadata only)

The 28 clips are excerpts of commercial films and are not included. `stimuli/stimuli.tsv` has one row per video index
(`stim_id` = `video_index` in the events = trigger value): source film, source clip database, targeted emotion and
valence (Chen et al. 2023, Supplementary Table S1), the Table S1 duration, and the presented duration measured in the
events of all 123 participants.

Facts from the events (video index + 101 onset to 102 offset, 123 x 28 spans):
- Every participant has all 28 spans; the median presented duration of every clip equals its Table S1 duration (rounded).
- Clip 22 (The Shawshank Redemption) exists in two versions: 83.92 s for sub-036 to sub-060 (25 participants) and 75.80 s
  for the other 98.
- 12 participants (sub-002, 008, 009, 028, 061-063, 071-075) have every span about 0.28 s longer (a playback/trigger latency,
  not different files).

Source databases (Table S1; column `source_database`): DCEF = Ge et al. 2018 (#1, #3), THU-EP = Hu et al. 2022 (#2),
FilmStim = Schaefer et al. 2010 (#4-7, #9, #13-16), PED = Hu et al. 2017/2019 (#10-12, #17-28); #8 has none.
DCEF and PED publish in/out timecodes for their clips. `database_clip` and `database_timecode` give them for the 10 clips
whose database has exactly one clip of the same film and the same whole-second duration (#1, #3, #10, #11, #19, #20, #21,
#24, #26, #28). Not listed: #23 (PED has two 34-s Totoro clips), #12, #17, #18, #25 (1 s apart), #22 (PED 83 s; FACED shows
75.80 s or 83.92 s), #27 (the PED Juno row is internally inconsistent). The timecodes are in the database authors' film
copies; PED notes they differ between versions. For #11 (Departures) both published points fall on shot changes, 0.10 s
and 0.53 s later, in a film-speed copy of the film; the span between the two shot changes is 60.44 s, while 60.15 s was
presented.

FilmStim clips (added 2026-10-09; columns `database_clip`, `database_position`): FilmStim distributes its clip files at
sites.uclouvain.be/ipsp/FilmStim (`En/<number>.mp4` = 25/24 film-speed versions of the PAL originals, `HiRes/<number>.mpeg`
= PAL), and the Hugging Face dataset Eureka-Leo/Emotion.Intelligence (`level3/.../FilmStim/videos/<number>`) mirrors them
byte for byte (five files compared; the LFS hashes of the other five equal the official files). `database_clip` names the
FilmStim number / code, file and SHA-256 for #4-7, #9 and #13-15 (Table S1 film + FilmStim scene list; #16, a fourth Blue
excerpt, has no FilmStim counterpart). The presented durations of #4, #5, #7 and #9 match the film-speed En files, not the
PAL HiRes ones (4% shorter). `database_position` says where the presented clip lies inside the FilmStim file, from a
frame-by-frame comparison of a cut of the same scene with the file: #4 Trainspotting 0.485-79.131 s of En/35 (both ends on
shot changes), #5 Indiana Jones 28.629-98.031 s of En/31 (both ends on shot changes; the only pair of shot changes 69.4 s
apart in the file), #6 Hellraiser the whole HiRes/57 (90.48 s; FACED 90.55 s). #7 The Shining: En/28 has three pairs of
shot changes 56.1 s apart (9.9, 104.2, 207.4 s), so no position is given. #9 The Exorcist and #13-15 Blue: the FACED clips
(105.8 / 35.1 / 44.3 / 38.9 s) are longer than the FilmStim files (104.9 / 16.2 / 40.2 / 25.3 s), so FACED cut them from
the films, not from the FilmStim files; #9 starts at the same frame as En/24 (within 0.05 s), the Blue starts are not
established. These positions are likely, not verified against FACED's own clip files.

Locating the clips in their source films: the exact trims are not published (the clip files are not on Synapse). Five
clips could be placed with two independent anchors (likely, not verified against FACED's own files):
- 4 Trainspotting: the FilmStim file En/35 from 0.485 to 79.131 s (see above).
- 5 Indiana Jones and the Last Crusade: the FilmStim file En/31 from 28.629 to 98.031 s (see above).
- 6 Hellraiser: the whole FilmStim file HiRes/57 (90.48 s at 25 fps; FACED 90.55 s).
- 9 The Exorcist: starts where the FilmStim file En/24 (104.94 s) starts; FACED's clip is 0.86 s longer (cut from the film).
- 23 My Neighbor Totoro: the girls exploring the new house, 31.8 s after the PED timecode 0:05:12.
Clip 22 (The Shawshank Redemption) is no longer listed: an earlier version of this section placed both versions at one
start 3.9 s before the PED in point, but the 83.92-s version fits the PED clip better (it starts at the PED in point), so
the start of the 75.80-s version is not established.
