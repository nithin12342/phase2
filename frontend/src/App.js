import React, { useState, useRef } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // Form state
  const [formData, setFormData] = useState({
    gender: '',
    country: '',
    occupation: '',
    days_indoors: '',
    is_self_employed: 'No',
    self_employed_date: '',
    growing_stress: '',
    changes_habits: '',
    mental_health_history: '',
    family_history: '',
    treatment_sought: '',
    mood_swings: '',
    work_interest: '',
    social_weakness: '',
    coping_struggles: '',
    interview_attended: '',
    care_options_awareness: ''
  });

  // File state
  const [photo, setPhoto] = useState(null);
  const [audio, setAudio] = useState(null);
  const [video, setVideo] = useState(null);
  const [doc, setDoc] = useState(null);

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState(null);
  const [error, setError] = useState(null);

  const photoRef = useRef(null);
  const audioRef = useRef(null);
  const videoRef = useRef(null);
  const docRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e, setter) => {
    if (e.target.files && e.target.files[0]) {
      setter(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSubmitResult(null);

    try {
      const formPayload = new FormData();
      formPayload.append('survey_data', JSON.stringify(formData));

      if (photo) formPayload.append('photo', photo);
      if (audio) formPayload.append('audio', audio);
      if (video) formPayload.append('video', video);
      if (doc) formPayload.append('doc', doc);

      const response = await fetch(`${API_URL}/api/v1/submit-survey`, {
        method: 'POST',
        body: formPayload
      });

      if (!response.ok) {
        let errorMessage = 'Submission failed';
        try {
          const errData = await response.json();
          errorMessage = errData.detail || errorMessage;
        } catch {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setSubmitResult(result);
    } catch (err) {
      console.error("Fetch error details:", err);
      setError(`Failed to connect to API at ${API_URL}. Error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Mental Health Assessment Survey</h1>
        <p>Multimodal Deep Learning for Depression Detection</p>
      </header>

      <form onSubmit={handleSubmit} className="survey-form">
        {/* Personal Information Section */}
        <section className="form-section">
          <h2>Personal Information</h2>
          <div className="form-grid">
            <div className="form-group">
              <label>Gender</label>
              <select name="gender" value={formData.gender} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label>Country</label>
              <input type="text" name="country" value={formData.country} onChange={handleInputChange} placeholder="e.g., India" />
            </div>
            <div className="form-group">
              <label>Occupation</label>
              <input type="text" name="occupation" value={formData.occupation} onChange={handleInputChange} placeholder="e.g., Student" />
            </div>
            <div className="form-group">
              <label>Days Spent Indoors</label>
              <select name="days_indoors" value={formData.days_indoors} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="1-14 days">1-14 days</option>
                <option value="15-30 days">15-30 days</option>
                <option value="31-60 days">31-60 days</option>
                <option value="More than 2 months">More than 2 months</option>
              </select>
            </div>
          </div>
        </section>

        {/* Employment Section */}
        <section className="form-section">
          <h2>Employment Status</h2>
          <div className="form-grid">
            <div className="form-group">
              <label>Self Employed?</label>
              <select name="is_self_employed" value={formData.is_self_employed} onChange={handleInputChange}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
            {formData.is_self_employed === 'Yes' && (
              <div className="form-group">
                <label>Since When?</label>
                <input type="date" name="self_employed_date" value={formData.self_employed_date} onChange={handleInputChange} />
              </div>
            )}
          </div>
        </section>

        {/* Mental Health Section */}
        <section className="form-section">
          <h2>Mental Health Indicators</h2>
          <div className="form-grid">
            <div className="form-group">
              <label>Growing Stress?</label>
              <select name="growing_stress" value={formData.growing_stress} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="Maybe">Maybe</option>
              </select>
            </div>
            <div className="form-group">
              <label>Changes in Habits?</label>
              <select name="changes_habits" value={formData.changes_habits} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="Maybe">Maybe</option>
              </select>
            </div>
            <div className="form-group">
              <label>Mental Health History?</label>
              <select name="mental_health_history" value={formData.mental_health_history} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="form-group">
              <label>Family History?</label>
              <select name="family_history" value={formData.family_history} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="form-group">
              <label>Treatment Sought?</label>
              <select name="treatment_sought" value={formData.treatment_sought} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="form-group">
              <label>Mood Swings</label>
              <select name="mood_swings" value={formData.mood_swings} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>
            <div className="form-group">
              <label>Work Interest</label>
              <select name="work_interest" value={formData.work_interest} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="Maybe">Maybe</option>
              </select>
            </div>
            <div className="form-group">
              <label>Social Weakness?</label>
              <select name="social_weakness" value={formData.social_weakness} onChange={handleInputChange}>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>
        </section>

        {/* Media Uploads Section */}
        <section className="form-section">
          <h2>Media Uploads</h2>
          <p className="section-hint">Upload files for multimodal analysis</p>
          <div className="form-grid">
            <div className="form-group file-upload">
              <label>Photo</label>
              <input type="file" ref={photoRef} accept="image/*" onChange={(e) => handleFileChange(e, setPhoto)} />
              {photo && <span className="file-name">{photo.name}</span>}
            </div>
            <div className="form-group file-upload">
              <label>Audio Recording</label>
              <input type="file" ref={audioRef} accept="audio/*" onChange={(e) => handleFileChange(e, setAudio)} />
              {audio && <span className="file-name">{audio.name}</span>}
            </div>
            <div className="form-group file-upload">
              <label>Video Recording</label>
              <input type="file" ref={videoRef} accept="video/*" onChange={(e) => handleFileChange(e, setVideo)} />
              {video && <span className="file-name">{video.name}</span>}
            </div>
            <div className="form-group file-upload">
              <label>Document / Notes</label>
              <input type="file" ref={docRef} accept=".pdf,.doc,.docx,.txt" onChange={(e) => handleFileChange(e, setDoc)} />
              {doc && <span className="file-name">{doc.name}</span>}
            </div>
          </div>
        </section>

        {/* Submit Button */}
        <div className="submit-section">
          <button type="submit" className="submit-btn" disabled={isSubmitting}>
            {isSubmitting ? 'Submitting...' : 'Submit Survey'}
          </button>
        </div>

        {/* Result/Error Display */}
        {error && (
          <div className="result-box error">
            <strong>Error:</strong> {error}
          </div>
        )}
        {submitResult && (
          <div className="result-box success">
            {submitResult.depression_risk && (
              <div className="risk-display" style={{ whiteSpace: 'pre-wrap' }}>
                {submitResult.depression_risk}
              </div>
            )}
          </div>
        )}
      </form>

      <footer className="app-footer">
        <p>Final Year Project - Multimodal Deep Learning API</p>
      </footer>
    </div>
  );
}

export default App;