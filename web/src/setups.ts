// C5 — the 11 one-click SIH demo setups (implementation plan R7 + GAP-4).
// Pure data; consumed by the Console setups gallery. Each names the B-patch
// it exercises so the judge can see which reliability fix the demo proves.

export interface Setup {
  id: string
  title: string
  task: string            // expected_task (routing contract)
  sampleNames: string[]
  query: string
  patches: string[]       // which B-patches this setup exercises
}

export const SETUPS: Setup[] = [
  { id: 'urban-growth', title: 'Urban growth', task: 'change_vqa',
    sampleNames: ['demo_change_2020.tif', 'demo_change_2024.tif'],
    query: 'Has the built-up area increased, decreased, or remained unchanged?',
    patches: ['B5 transitions'] },
  { id: 'flood-water', title: 'Flood / water extent', task: 'grounding',
    sampleNames: ['demo_single_multispectral.tif'],
    query: 'Highlight the water body in this image.',
    patches: ['B3 ranked regions'] },
  { id: 'land-cover', title: 'Land-cover brief', task: 'captioning',
    sampleNames: ['demo_single_multispectral.tif'],
    query: 'Describe the land-cover of this image.',
    patches: ['B6 query caption'] },
  { id: 'road-check', title: 'Road check', task: 'single_vqa',
    sampleNames: ['demo_single_optical.png'],
    query: 'Is there a road?',
    patches: ['B9 traceable answer'] },
  { id: 'sar-night', title: 'SAR night', task: 'optical_sar',
    sampleNames: ['demo_isroformat_optical.tif', 'demo_isroformat_sar.tif'],
    query: 'Use the optical and SAR image together.',
    patches: ['B1 modality certainty', 'B4 agreement map'] },
  { id: 'change-qa', title: 'Change Q+A', task: 'change_vqa',
    sampleNames: ['demo_change_2020.tif', 'demo_change_2024.tif'],
    query: 'What changed between these two dates?',
    patches: ['B5 transitions'] },
  { id: 'crop-veg', title: 'Crop / vegetation', task: 'captioning',
    sampleNames: ['demo_single_multispectral.tif'],
    query: 'What crops or vegetation are present?',
    patches: ['B6 query caption'] },
  { id: 'investigation', title: 'Full investigation', task: 'investigation',
    sampleNames: ['demo_change_2020.tif', 'demo_change_2024.tif'],
    query: 'Investigate the change around the water body.',
    patches: ['B5 transitions', 'B7 calibration'] },
  { id: 'counting', title: 'Counting stress', task: 'single_vqa',
    sampleNames: ['demo_single_optical.png'],
    query: 'How many buildings are there?',
    patches: ['C4 honesty banner'] },
  { id: 'isro-pair', title: 'ISRO-format pair', task: 'optical_sar',
    sampleNames: ['demo_pair_optical.tif', 'demo_pair_sar.tif'],
    query: 'Use the optical and SAR image together.',
    patches: ['B1 modality certainty', 'B4 agreement map', 'B8 geo'] },
  { id: 'change-desc', title: 'Change description', task: 'change_analysis',
    sampleNames: ['demo_change_2020.tif', 'demo_change_2024.tif'],
    query: 'Describe the change between these two dates.',
    patches: ['B5 transitions', 'B9 traceable answer'] },
]
