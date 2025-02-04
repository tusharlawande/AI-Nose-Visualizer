import { useState, useCallback } from 'react'
import './App.css'

const ENHANCEMENT_STYLES = {
  natural: {
    name: 'Natural',
    description: 'Subtle enhancement that maintains a very natural appearance',
    icon: '🍃'
  },
  refined: {
    name: 'Refined',
    description: 'Enhanced definition with balanced sophistication',
    icon: '✨'
  },
  elegant: {
    name: 'Elegant',
    description: 'Sophisticated enhancement with subtle contouring',
    icon: '💫'
  },
  sculpted: {
    name: 'Sculpted',
    description: 'Dramatic enhancement with defined contours',
    icon: '🗿'
  },
  conservative: {
    name: 'Conservative',
    description: 'Minimal enhancement for subtle refinement',
    icon: '🪶'
  },
  balanced: {
    name: 'Balanced',
    description: 'Harmonious blend of enhancement and naturalness',
    icon: '☯️'
  }
}

const VIEW_TYPES = {
  frontView: 'Front View',
  sideView: 'Side View',
  threeQuarter: 'Three Quarter View'
}

function App() {
  const [viewType, setViewType] = useState('frontView')
  const [style, setStyle] = useState('natural')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [enhancedImage, setEnhancedImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleDrop = useCallback(async (e) => {
    e.preventDefault()
    setError(null)
    
    const droppedFile = e.dataTransfer?.files[0] || e.target.files[0]
    
    if (!droppedFile) {
      setError('Please select an image file')
      return
    }
    
    if (!droppedFile.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }
    
    setFile(droppedFile)
    
    // Create preview
    const reader = new FileReader()
    reader.onload = () => setPreview(reader.result)
    reader.readAsDataURL(droppedFile)
    
    // Process image
    await processImage(droppedFile)
  }, [])

  const processImage = async (imageFile) => {
    try {
      setLoading(true)
      setError(null)
      setEnhancedImage(null)
      
      const formData = new FormData()
      formData.append('file', imageFile)
      formData.append('view_type', viewType)
      formData.append('style', style)

      const response = await fetch('http://localhost:8003/process-image/', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to process image')
      }

      const blob = await response.blob()
      const imageUrl = URL.createObjectURL(blob)
      setEnhancedImage(imageUrl)
    } catch (err) {
      console.error('Error processing image:', err)
      setError(err.message || 'Failed to process image. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleStyleChange = useCallback((newStyle) => {
    setStyle(newStyle)
    if (file) {
      processImage(file)
    }
  }, [file])

  const handleViewTypeChange = useCallback((e) => {
    setViewType(e.target.value)
    if (file) {
      processImage(file)
    }
  }, [file])

  const handleTryAgain = useCallback(() => {
    setError(null)
    setEnhancedImage(null)
    setFile(null)
    setPreview(null)
  }, [])

  return (
    <div className="app-container">
      <header className="header">
        <h1>Advanced Nose Visualization Technology</h1>
        <h2>Experience the future of facial enhancement visualization</h2>
      </header>

      <div className="enhancement-features">
        <div className="feature-card">
          <span className="icon">🎯</span>
          <h3>Precision Enhancement</h3>
          <p>Advanced algorithms for accurate visualization</p>
        </div>
        <div className="feature-card">
          <span className="icon">🎨</span>
          <h3>Multiple Styles</h3>
          <p>Choose from various enhancement styles</p>
        </div>
        <div className="feature-card">
          <span className="icon">⚡</span>
          <h3>Real-time Processing</h3>
          <p>Instant visualization of enhancements</p>
        </div>
      </div>

      <div className="controls">
        <div className="control-group">
          <label htmlFor="viewType">View Type</label>
          <select 
            id="viewType" 
            value={viewType} 
            onChange={handleViewTypeChange}
          >
            {Object.entries(VIEW_TYPES).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="style-preview">
        {Object.entries(ENHANCEMENT_STYLES).map(([styleKey, styleData]) => (
          <div 
            key={styleKey}
            className={`style-option ${style === styleKey ? 'selected' : ''}`}
            onClick={() => handleStyleChange(styleKey)}
          >
            <span className="style-icon">{styleData.icon}</span>
            <h4>{styleData.name}</h4>
            <p>{styleData.description}</p>
          </div>
        ))}
      </div>

      <div 
        className="dropzone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleDrop}
          style={{ display: 'none' }}
        />
        <div className="upload-prompt">
          <img src="/upload-icon.svg" alt="Upload" className="upload-icon" />
          <p>Drag & Drop or Click to Upload</p>
          <small>Supports JPG, PNG, and WEBP</small>
        </div>
      </div>

      {error && (
        <div className="status-message error">
          <p>{error}</p>
          <button className="try-again-button" onClick={handleTryAgain}>
            Try Again
          </button>
        </div>
      )}

      {loading && (
        <div className="status-message loading">
          <div className="spinner"></div>
          <p>Enhancing your image...</p>
        </div>
      )}

      {(preview || enhancedImage) && (
        <div className="results-container">
          {preview && (
            <div className="image-container">
              <h3>Original Image</h3>
              <img src={preview} alt="Original" className="result-image" />
            </div>
          )}
          {enhancedImage && (
            <div className="image-container">
              <h3>Enhanced Image</h3>
              <img src={enhancedImage} alt="Enhanced" className="result-image" />
              <button 
                className="download-button"
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = enhancedImage;
                  link.download = 'enhanced-nose.png';
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                }}
              >
                <span className="download-icon">⬇️</span>
                Download Enhanced Image
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
