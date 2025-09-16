const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json()); // needed for POST requests

// Serve static files (images) from the backend's static directory
// app.use('/static', express.static(path.join(__dirname, 'static')));

// Proxy for summary
app.get('/api/data', async (req, res) => {
  try {
    const response = await fetch('http://localhost:8000/api/data');
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error proxying /api/data:', error);
    res.status(500).json({ error: 'Failed to fetch data from backend.' });
  }
});

// Proxy for chat
app.post('/api/chat', async (req, res) => {
  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    
    // Log the response type for debugging
    console.log('Backend response type:', data.type);
    
    res.json(data);
  } catch (error) {
    console.error('Error proxying /api/chat:', error);
    res.status(500).json({ error: 'Failed to fetch chat response from backend.' });
  }
});

// Proxy for static files from backend
app.get('/static/:filename', async (req, res) => {
  try {
    const { filename } = req.params;
    const response = await fetch(`http://localhost:8000/static/${filename}`);
    
    if (!response.ok) {
      return res.status(404).json({ error: 'Image not found' });
    }
    
    // Set appropriate content type
    res.set('Content-Type', response.headers.get('content-type'));
    
    // Pipe the image data
    response.body.pipe(res);
  } catch (error) {
    console.error('Error proxying static file:', error);
    res.status(500).json({ error: 'Failed to fetch image from backend.' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'ARGO Frontend Proxy Server'
  });
});

// Serve React frontend build
app.use(express.static(path.join(__dirname, '../frontend/build')));

// Catch-all handler for React Router
app.get('/*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`✅ Express proxy server running on http://localhost:${PORT}`);
  console.log(`📊 Serving ARGO chatbot frontend with plot visualization support`);
  console.log(`🖼️  Static images available at: http://localhost:${PORT}/static/[filename]`);
});
