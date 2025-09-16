import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { MessageSquare, Send, Globe, Activity, Thermometer, Droplets, Navigation, MapPin, Layers, Maximize2, Minimize2, RefreshCw, Download, Settings, Menu, X, Search, Satellite, Waves, TrendingUp, AlertCircle } from 'lucide-react';

const FloatChat = () => {
  // Chat state
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: 'Welcome to FloatChat! I can help you explore ARGO ocean data in real-time. Try asking about temperature trends, salinity profiles, or click the Indian Ocean Map to see live float positions.',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mapModalOpen, setMapModalOpen] = useState(false);
  const [selectedFloat, setSelectedFloat] = useState(null);
  
  // Real-time variables
  const [realTimeData, setRealTimeData] = useState({
    activeFloats: 4127,
    lastUpdate: new Date(),
    avgTemperature: 15.2,
    avgSalinity: 34.7,
    dataPoints: 2847592,
    connectedRegions: 8,
    systemStatus: 'operational'
  });

  // Indian Ocean ARGO floats with real-time simulation
  const [indianOceanFloats, setIndianOceanFloats] = useState([
    { id: 'IN001', lat: -10.5, lon: 72.3, temp: 28.4, salinity: 34.2, status: 'active', lastUpdate: '2 min ago', depth: 0 },
    { id: 'IN002', lat: -15.2, lon: 68.7, temp: 26.8, salinity: 34.5, status: 'active', lastUpdate: '5 min ago', depth: 150 },
    { id: 'IN003', lat: -8.9, lon: 80.1, temp: 29.1, salinity: 33.9, status: 'active', lastUpdate: '1 min ago', depth: 0 },
    { id: 'IN004', lat: -22.4, lon: 75.6, temp: 24.2, salinity: 35.1, status: 'maintenance', lastUpdate: '2 hours ago', depth: 300 },
    { id: 'IN005', lat: -12.7, lon: 85.3, temp: 27.6, salinity: 34.3, status: 'active', lastUpdate: '3 min ago', depth: 50 },
    { id: 'IN006', lat: -18.1, lon: 90.2, temp: 25.9, salinity: 34.8, status: 'active', lastUpdate: '1 min ago', depth: 200 },
    { id: 'IN007', lat: -5.3, lon: 95.4, temp: 30.2, salinity: 33.7, status: 'active', lastUpdate: '4 min ago', depth: 0 },
    { id: 'IN008', lat: -25.8, lon: 82.7, temp: 22.8, salinity: 35.3, status: 'active', lastUpdate: '2 min ago', depth: 500 }
  ]);

  const chatEndRef = useRef(null);

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => ({
        ...prev,
        activeFloats: prev.activeFloats + Math.floor(Math.random() * 3) - 1,
        lastUpdate: new Date(),
        avgTemperature: prev.avgTemperature + (Math.random() - 0.5) * 0.1,
        avgSalinity: prev.avgSalinity + (Math.random() - 0.5) * 0.05,
        dataPoints: prev.dataPoints + Math.floor(Math.random() * 100) + 50
      }));

      // Update float data
      setIndianOceanFloats(prev => prev.map(float => ({
        ...float,
        temp: float.temp + (Math.random() - 0.5) * 0.3,
        salinity: float.salinity + (Math.random() - 0.5) * 0.1,
        lastUpdate: Math.random() < 0.3 ? 'Just now' : float.lastUpdate
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Python backend communication functions (commented out to avoid ESLint warning)
  // const sendToPythonBackend = async (endpoint, data) => {
  //   try {
  //     // This would connect to your Python backend
  //     const response = await fetch(`/api/${endpoint}`, {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify(data),
  //     });
  //     return await response.json();
  //   } catch (error) {
  //     console.log('Python backend connection simulated:', { endpoint, data });
  //     // Simulate response for demo
  //     return { success: true, data: 'Simulated Python response' };
  //   }
  // };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: currentInput })
  });

  const data = await response.json();
  
  // Handle different response types (text, image, error)
  let messageContent;
  let messageType = "text";
  
  if (data.type === "image") {
    // Handle image/graph response
    messageContent = (
      <div className="graph-container">
        <img 
          src={data.url || `data:image/png;base64,${data.base64}`} 
          alt={data.description || "Graph"} 
          className="w-full rounded-lg border border-slate-600/50 shadow-lg"
        />
        <div className="mt-2 text-sm text-slate-300">
          {data.description} ({data.data_points} data points)
        </div>
      </div>
    );
    messageType = "image";
  } else {
    // Handle text response
    messageContent = data.content || data.response || "⚠️ No response from backend.";
  }

  const assistantMessage = {
    id: Date.now() + 1,
    type: "assistant",
    content: messageContent,
    contentType: messageType,
    timestamp: new Date()
  };

  setMessages(prev => [...prev, assistantMessage]);
      
      // If it was a graph request, log success
      if (messageType === "image") {
        console.log("Graph displayed successfully");
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: "assistant",
        content: "❌ Error connecting to backend.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickActions = [
    'Show current Indian Ocean temperature data',
    'Display active ARGO floats near India',
    'Show salinity vs temperature graph',
    'Show temperature vs depth graph',
    'Show salinity vs depth graph',
    'Show pressure vs time graph',
    'Show pH vs salinity graph',
    'Analyze salinity trends in monsoon regions',
    'Find temperature anomalies in the Bay of Bengal'
  ];

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white overflow-hidden">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-slate-800/40 backdrop-blur-xl border-r border-slate-700/50`}>
        <div className="p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-xl flex items-center justify-center">
              <Waves className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
                FloatChat
              </h1>
              <p className="text-xs text-slate-400">Real-time Ocean Data AI</p>
            </div>
          </div>

          {/* Real-time Status Dashboard */}
          <div className="bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-xl p-4 mb-6 border border-slate-600/30">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-cyan-300">System Status</h3>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400">Live</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Activity className="w-3 h-3 text-cyan-400" />
                  <span className="text-xs text-slate-300">Active Floats</span>
                </div>
                <div className="text-lg font-bold text-white">{realTimeData.activeFloats.toLocaleString()}</div>
              </div>
              
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Thermometer className="w-3 h-3 text-orange-400" />
                  <span className="text-xs text-slate-300">Avg Temp</span>
                </div>
                <div className="text-lg font-bold text-white">{realTimeData.avgTemperature.toFixed(1)}°C</div>
              </div>
              
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Droplets className="w-3 h-3 text-blue-400" />
                  <span className="text-xs text-slate-300">Avg Salinity</span>
                </div>
                <div className="text-lg font-bold text-white">{realTimeData.avgSalinity.toFixed(1)} PSU</div>
              </div>
              
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <TrendingUp className="w-3 h-3 text-green-400" />
                  <span className="text-xs text-slate-300">Data Points</span>
                </div>
                <div className="text-lg font-bold text-white">{(realTimeData.dataPoints / 1000000).toFixed(1)}M</div>
              </div>
            </div>
            
            <div className="mt-3 text-xs text-slate-400">
              Last update: {realTimeData.lastUpdate.toLocaleTimeString()}
            </div>
          </div>

          {/* Indian Ocean Map Button */}
          <button
            onClick={() => setMapModalOpen(true)}
            className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 rounded-xl p-4 mb-6 transition-all duration-300 transform hover:scale-105 shadow-lg"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                <MapPin className="w-4 h-4" />
              </div>
              <div className="text-left">
                <div className="font-semibold">Indian Ocean Map</div>
                <div className="text-xs text-indigo-200">Live ARGO Float Positions</div>
              </div>
            </div>
          </button>

          {/* Quick Actions */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Quick Actions</h3>
            {quickActions.map((action, index) => (
              <button
                key={index}
                onClick={() => setInput(action)}
                className="w-full text-left p-3 text-sm bg-slate-700/20 hover:bg-slate-600/30 rounded-lg transition-all duration-200 border border-slate-600/20 hover:border-cyan-500/30 hover:shadow-lg"
              >
                {action}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 bg-slate-800/30 backdrop-blur-xl border-b border-slate-700/50">
          <div className="flex items-center space-x-4">
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <Satellite className="w-4 h-4 text-green-400" />
                <span className="text-sm">Connected to Python Backend</span>
              </div>
              <div className="flex items-center space-x-2">
                <Globe className="w-4 h-4 text-blue-400" />
                <span className="text-sm">{realTimeData.connectedRegions} Ocean Regions</span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button 
              onClick={() => setMapModalOpen(true)}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 rounded-lg transition-all duration-200 flex items-center space-x-2"
            >
              <MapPin className="w-4 h-4" />
              <span className="hidden sm:inline">Indian Ocean</span>
            </button>
            <div className="relative group">
              <button 
                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 rounded-lg transition-all duration-200 flex items-center space-x-2"
              >
                <Activity className="w-4 h-4" />
                <span className="hidden sm:inline">Generate Graph</span>
              </button>
              <div className="absolute left-0 mt-2 w-56 rounded-md shadow-lg bg-slate-800 ring-1 ring-black ring-opacity-5 focus:outline-none z-10 hidden group-hover:block">
                <div className="py-1" role="menu" aria-orientation="vertical">
                  <button onClick={() => setInput('Show salinity vs temperature graph')} className="block px-4 py-2 text-sm text-white hover:bg-slate-700 w-full text-left" role="menuitem">Salinity vs Temperature</button>
                  <button onClick={() => setInput('Show temperature vs depth graph')} className="block px-4 py-2 text-sm text-white hover:bg-slate-700 w-full text-left" role="menuitem">Temperature vs Depth</button>
                  <button onClick={() => setInput('Show salinity vs depth graph')} className="block px-4 py-2 text-sm text-white hover:bg-slate-700 w-full text-left" role="menuitem">Salinity vs Depth</button>
                  <button onClick={() => setInput('Show pressure vs time graph')} className="block px-4 py-2 text-sm text-white hover:bg-slate-700 w-full text-left" role="menuitem">Pressure vs Time</button>
                  <button onClick={() => setInput('Show pH vs salinity graph')} className="block px-4 py-2 text-sm text-white hover:bg-slate-700 w-full text-left" role="menuitem">pH vs Salinity</button>
                </div>
              </div>
            </div>
            <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
            <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl ${message.type === 'user' ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white' : 'bg-slate-800/50 border border-slate-700/50'} rounded-2xl p-4 shadow-lg`}>
                <div className="flex items-start space-x-3">
                  {message.type === 'assistant' && (
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                      <Waves className="w-4 h-4 text-white" />
                    </div>
                  )}
                  <div className="flex-1">
  {/* Render content based on contentType */}
  {message.contentType === 'image' ? (
    <div className="message-content">{message.content}</div>
  ) : (
    <p className="leading-relaxed">{message.content}</p>
  )}
  <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-600/30">
                      <span className="text-xs text-slate-400">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                      {message.type === 'assistant' && (
                        <div className="flex space-x-2">
                          <button className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors">
                            <Download className="w-3 h-3 inline mr-1" />
                            Export
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-4 shadow-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full flex items-center justify-center">
                    <Waves className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-slate-800/30 backdrop-blur-xl border-t border-slate-700/50">
          <div className="flex items-end space-x-4">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendMessage())}
                placeholder="Ask about real-time ocean data, ARGO float positions, or request analysis..."
                rows={2}
                className="w-full bg-slate-700/50 border border-slate-600/50 rounded-xl px-4 py-3 pr-12 text-white placeholder-slate-400 focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 resize-none"
              />
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                className="absolute bottom-3 right-3 p-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
            <span>Connected to Python Backend • Real-time ARGO Data</span>
            <span>Press Enter to send • Shift+Enter for new line</span>
          </div>
        </div>
      </div>

      {/* Indian Ocean Map Modal */}
      {mapModalOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-800 rounded-2xl w-full max-w-6xl h-5/6 flex flex-col shadow-2xl border border-slate-700">
            <div className="flex items-center justify-between p-6 border-b border-slate-700">
              <div className="flex items-center space-x-3">
                <MapPin className="w-6 h-6 text-purple-400" />
                <div>
                  <h2 className="text-xl font-bold">Indian Ocean ARGO Floats</h2>
                  <p className="text-sm text-slate-400">Real-time positions and data</p>
                </div>
              </div>
              <button 
                onClick={() => setMapModalOpen(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="flex-1 flex">
              {/* Map Area */}
              <div className="flex-1 relative bg-gradient-to-br from-blue-900/20 to-cyan-900/20">
                {/* Simulated Map Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-blue-800/30 to-cyan-700/30 rounded-bl-2xl">
                  <div className="absolute inset-0 opacity-20" style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
                  }}></div>
                </div>
                
                {/* Float Markers */}
                {indianOceanFloats.map((float, index) => (
                  <div
                    key={float.id}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                    style={{
                      left: `${((float.lon - 60) / 40) * 100}%`,
                      top: `${((15 - float.lat) / 40) * 100}%`
                    }}
                    onClick={() => setSelectedFloat(float)}
                  >
                    <div className={`w-4 h-4 rounded-full border-2 border-white shadow-lg animate-pulse ${
                      float.status === 'active' ? 'bg-green-400' : 'bg-yellow-400'
                    }`}></div>
                    <div className="absolute top-5 left-1/2 transform -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap opacity-0 hover:opacity-100 transition-opacity">
                      {float.id}
                    </div>
                  </div>
                ))}

                {/* Map Labels */}
                <div className="absolute top-4 left-4 bg-slate-800/80 rounded-lg p-3">
                  <h3 className="font-semibold text-cyan-300 mb-2">Indian Ocean Region</h3>
                  <div className="space-y-1 text-xs">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Active Float ({indianOceanFloats.filter(f => f.status === 'active').length})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                      <span>Maintenance ({indianOceanFloats.filter(f => f.status === 'maintenance').length})</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Float Details Sidebar */}
              <div className="w-80 bg-slate-700/30 p-6 border-l border-slate-700">
                <h3 className="font-semibold text-cyan-300 mb-4">ARGO Float Details</h3>
                
                {selectedFloat ? (
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-600/30">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-white">{selectedFloat.id}</h4>
                      <div className={`px-2 py-1 rounded-full text-xs ${
                        selectedFloat.status === 'active' ? 'bg-green-400/20 text-green-400' : 'bg-yellow-400/20 text-yellow-400'
                      }`}>
                        {selectedFloat.status}
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div>
                        <div className="text-xs text-slate-400">Position</div>
                        <div className="text-sm">{selectedFloat.lat.toFixed(2)}°, {selectedFloat.lon.toFixed(2)}°</div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <div className="text-xs text-slate-400">Temperature</div>
                          <div className="text-sm font-semibold text-orange-400">{selectedFloat.temp.toFixed(1)}°C</div>
                        </div>
                        <div>
                          <div className="text-xs text-slate-400">Salinity</div>
                          <div className="text-sm font-semibold text-blue-400">{selectedFloat.salinity.toFixed(1)} PSU</div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="text-xs text-slate-400">Current Depth</div>
                        <div className="text-sm">{selectedFloat.depth}m</div>
                      </div>
                      
                      <div>
                        <div className="text-xs text-slate-400">Last Update</div>
                        <div className="text-sm">{selectedFloat.lastUpdate}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-600/30 text-center">
                    <MapPin className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                    <p className="text-sm text-slate-400">Click on a float marker to view details</p>
                  </div>
                )}

                {/* Summary Stats */}
                <div className="mt-6 space-y-3">
                  <h4 className="font-semibold text-slate-300">Regional Summary</h4>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Average Temperature</div>
                    <div className="text-lg font-bold text-orange-400">
                      {(indianOceanFloats.reduce((sum, f) => sum + f.temp, 0) / indianOceanFloats.length).toFixed(1)}°C
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Average Salinity</div>
                    <div className="text-lg font-bold text-blue-400">
                      {(indianOceanFloats.reduce((sum, f) => sum + f.salinity, 0) / indianOceanFloats.length).toFixed(1)} PSU
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FloatChat;
