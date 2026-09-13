import { PatientInput } from '../types/prediction';

export interface ValidationError {
  field: keyof PatientInput;
  message: string;
}

export function validatePatientInput(data: PatientInput): ValidationError[] {
  const errors: ValidationError[] = [];

  if (data.age < 18 || data.age > 120) {
    errors.push({ field: 'age', message: 'Age must be between 18 and 120 years.' });
  }

  if (data.resting_bp < 70 || data.resting_bp > 260) {
    errors.push({ field: 'resting_bp', message: 'Resting Blood Pressure must be between 70 and 260 mmHg.' });
  }

  if (data.cholesterol < 80 || data.cholesterol > 700) {
    errors.push({ field: 'cholesterol', message: 'Serum Cholesterol must be between 80 and 700 mg/dL.' });
  }

  if (data.max_heart_rate < 50 || data.max_heart_rate > 250) {
    errors.push({ field: 'max_heart_rate', message: 'Max Heart Rate must be between 50 and 250 bpm.' });
  }

  if (data.oldpeak < -5.0 || data.oldpeak > 10.0) {
    errors.push({ field: 'oldpeak', message: 'ST Depression (Oldpeak) must be between -5.0 and 10.0 mm.' });
  }

  if (data.major_vessels < 0 || data.major_vessels > 3) {
    errors.push({ field: 'major_vessels', message: 'Major Vessels count must be between 0 and 3.' });
  }

  return errors;
}
