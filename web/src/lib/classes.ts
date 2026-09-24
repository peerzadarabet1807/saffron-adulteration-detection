import classNames from '../data/class_names.json';

export type ClassId = 'C1' | 'C2' | 'C3' | 'C4' | 'C5';
export type Group = 'pure' | 'counterfeit';

export interface ClassInfo {
  id: ClassId;
  group: Group;
  /** short label, used in bars and tables */
  name: string;
  /** one-sentence description */
  description: string;
}

export const CLASS_INFO: Record<ClassId, ClassInfo> = {
  C1: {
    id: 'C1',
    group: 'pure',
    name: 'Pure, dryer-dried',
    description: 'Genuine saffron stigma dried with a dryer.',
  },
  C2: {
    id: 'C2',
    group: 'pure',
    name: 'Pure, press-dried',
    description: 'Genuine saffron stigma dried by pressing.',
  },
  C3: {
    id: 'C3',
    group: 'counterfeit',
    name: 'Dried saffron stem',
    description: 'The pale style (stem) of the saffron flower, sold as if it were stigma.',
  },
  C4: {
    id: 'C4',
    group: 'counterfeit',
    name: 'Dyed saffron stem',
    description: 'Saffron stem coloured red with food dye to imitate stigma.',
  },
  C5: {
    id: 'C5',
    group: 'counterfeit',
    name: 'Artificial saffron',
    description: 'Synthetic threads made to look like saffron.',
  },
};

export const GROUP_LABEL: Record<Group, string> = { pure: 'Pure', counterfeit: 'Counterfeit' };

/** Model output order (models/class_names.json). */
export const CLASS_ORDER = classNames as ClassId[];

if (CLASS_ORDER.length !== 5 || CLASS_ORDER.some((c) => !(c in CLASS_INFO))) {
  throw new Error(`class_names.json does not match the known classes: ${CLASS_ORDER.join(', ')}`);
}
