const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json()); // needed for POST requests

// Proxy for summary
app.get('/api/data', async (req, res) => {
  try {
    const response = await fetch('http://localhost:5000/api/data');
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
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error proxying /api/chat:', error);
    res.status(500).json({ error: 'Failed to fetch chat response from backend.' });
  }
});

// Serve React frontend build
app.use(express.static(path.join(__dirname, '../frontend/build')));

app.listen(PORT, () => {
  console.log(`✅ Express proxy server running on http://localhost:${PORT}`);
});
