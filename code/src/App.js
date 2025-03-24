import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const NETWORK_CONFIG = {
  MAX_RETRIES: 5,
  BASE_RETRY_DELAY: 3000,
  MAX_CONCURRENT: 3,
  OFFLINE_CHECK_INTERVAL: 5000,
  REQUEST_TIMEOUT: 15000
};

const requestQueue = [];
let activeRequests = new Set();
let isOnline = navigator.onLine;

const processQueue = async () => {
  while (requestQueue.length > 0 && 
         activeRequests.size < NETWORK_CONFIG.MAX_CONCURRENT && 
         isOnline) {
    const { url, options, resolve, reject, retryCount = 0 } = requestQueue.shift();
    const controller = new AbortController();

    try {
      activeRequests.add(controller);
      const response = await axios({
        ...options,
        url,
        signal: controller.signal,
        timeout: NETWORK_CONFIG.REQUEST_TIMEOUT
      });
      resolve(response.data);
    } catch (err) {
      if (retryCount < NETWORK_CONFIG.MAX_RETRIES && 
          !axios.isCancel(err) && 
          (err.code === 'ECONNABORTED' || !err.response)) {
        const delay = Math.min(NETWORK_CONFIG.BASE_RETRY_DELAY * Math.pow(2, retryCount), 30000);
        setTimeout(() => {
          requestQueue.push({ url, options, resolve, reject, retryCount: retryCount + 1 });
          processQueue();
        }, delay);
      } else {
        reject(err.response 
          ? `Server: ${err.response.status} - ${err.response.data?.detail || 'Unknown error'}` 
          : !isOnline ? 'Network offline' : err.message);
      }
    } finally {
      activeRequests.delete(controller);
    }
  }
};

const App = () => {
  const [customerId, setCustomerId] = useState('');
  const [businessId, setBusinessId] = useState('');
  const [query, setQuery] = useState('');
  const [chatbotResponse, setChatbotResponse] = useState('');
  const [financialAdvice, setFinancialAdvice] = useState('');
  const [loanApproval, setLoanApproval] = useState('');
  const [fraudRisk, setFraudRisk] = useState('');
  const [churnRisk, setChurnRisk] = useState('');
  const [wealthAdvice, setWealthAdvice] = useState('');
  const [subscriptionAdvice, setSubscriptionAdvice] = useState('');
  const [marketInsights, setMarketInsights] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [classificationResult, setClassificationResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [networkStatus, setNetworkStatus] = useState(isOnline ? 'online' : 'offline');
  const abortController = useRef(new AbortController());
  const [listening, setListening] = useState(false);  
  
  const recognitionRef = useRef(null); 
  const [voiceEnabled, setVoiceEnabled] = useState(true); 
  const [micEnabled, setMicEnabled] = useState(true); 
  const [customerDetails, setCustomerDetails] = useState(null);

  const [fraudData, setFraudData] = useState({
    isLoading: false,
    error: null,
    result: null
  });


  

  useEffect(() => {
    const handleOnline = () => {
      isOnline = true;
      setNetworkStatus('online');
      processQueue();
    };
    const handleOffline = () => {
      isOnline = false;
      setNetworkStatus('offline');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    const interval = setInterval(() => {
      setNetworkStatus(navigator.onLine ? 'online' : 'offline');
    }, NETWORK_CONFIG.OFFLINE_CHECK_INTERVAL);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(interval);
      abortController.current.abort();
    };
  }, []);

// ... existing code ...

const handleApiError = (err, context) => {
  if (axios.isCancel(err)) return;
  
  let errorMessage = '';
  
  if (err.response) {
    // Handle specific HTTP error codes
    switch (err.response.status) {
      case 404:
        errorMessage = `${context}: Invalid ID or resource not found. Please check your input.`;
        break;
      case 400:
        errorMessage = `${context}: Please provide valid input values.`;
        break;
      case 401:
        errorMessage = `${context}: Authentication required. Please log in.`;
        break;
      case 403:
        errorMessage = `${context}: Access denied. Please check permissions.`;
        break;
      default:
        errorMessage = `${context}: ${err.response.data?.detail || 'An unexpected error occurred'}`;
    }
  } else if (!navigator.onLine) {
    errorMessage = `${context}: No internet connection - request will be retried automatically.`;
  } else if (err.message.includes('Network Error')) {
    errorMessage = `${context}: Unable to connect to server - please try again later.`;
  } else {
    errorMessage = `${context}: Please check your input values and try again.`;
  }
  
  setError(errorMessage);
};

// ... rest of the code ...

  const clearAll = () => {
    if (window.confirm("Clear all inputs and responses?")) {
      abortController.current.abort();
      activeRequests.forEach(c => c.abort());
      activeRequests.clear();
      abortController.current = new AbortController();
      setCustomerId('');
      setBusinessId('');
      setQuery('');
      setChatbotResponse('');
      setFinancialAdvice('');
      setLoanApproval('');
      setFraudRisk('');
      setChurnRisk('');
      setWealthAdvice('');
      setSubscriptionAdvice('');
      setMarketInsights('');
      setError('');
      setClassificationResult(null);
      setSelectedFile(null);
    }
  };

  const queuedRequest = async (url, options = {}) => {
    return new Promise((resolve, reject) => {
      requestQueue.push({
        url: `${url}?_=${Date.now()}`,
        options: { ...options, signal: abortController.current.signal },
        resolve,
        reject
      });
      processQueue();
    });
  };

  const fetchChatbotResponse = async (query, customerId = null) => {
    try {
        console.log("📢 Sending API request:", { query, customerId });

        // ✅ Construct the payload dynamically
        const payload = { query };
        if (customerId && customerId.trim() !== "") payload.customer_id = customerId;  // ✅ Add only if valid

        const response = await fetch("http://127.0.0.1:8000/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(payload)  // ✅ Only sends customer_id if provided
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`🚨 Server Error ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        console.log("✅ Chatbot Response:", data);

        return data;

    } catch (error) {
        console.error("❌ Chatbot API Error:", error.message);
        return { response: "⚠️ AI service is temporarily unavailable. Please try again later." };
    }
};




const fetchFinancialAdvice = async () => {
  setIsLoading(true);
  setError('');
  try {
      if (!customerId) {
          setError('Customer ID is required');
          return;
      }

      const data = await queuedRequest(`http://127.0.0.1:8000/financial/${customerId}`);

      if (data.status === "success") {
          if (data.message) {
              // Customer not found case
              setFinancialAdvice(data.message);
              setCustomerDetails(null);  // Clear previous details
          } else {
              // Customer found case - Extract and update details
              setFinancialAdvice(data.practical_unique_financial_advice || "No advice available.");
              
              // Update customer details
              setCustomerDetails({
                  age: data.age,
                  income: data.income,
                  creditScore: data.credit_score
              });
          }
      } else {
          throw new Error(data.message || "Unknown error occurred");
      }
  } catch (err) {
      handleApiError(err, 'Fetching financial advice');
  } finally {
      setIsLoading(false);
  }
};

 
  const fetchDocumentClassification = async () => {
    setIsLoading(true);
    setError('');
    try {
      if (!selectedFile) {
        setError('Please select an image file first');
        return;
      }
      
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      const response = await fetch('http://127.0.0.1:8000/classify-image', {
        method: 'POST',
        body: formData,
      });
  
      console.log('Raw Response:', response); // Debugging
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server Error:', errorText); // Debugging
        throw new Error(`Server error: ${response.status}`);
      }
  
      const data = await response.json();
      console.log('Parsed Data:', data); // Debugging
      
      const result = {
        documentAnalysis: {
          category: data.document_analysis?.category || 'Unknown Category',
          match: data.document_analysis?.match || 'No Match',
          confidence: data.document_analysis?.confidence 
            ? `${Math.round(data.document_analysis.confidence * 100)}%`
            : '0%'
        },
        extractedText: data.extracted_text || 'No text extracted',
        categorizedSpending: data.categorized_spending || {},
        financialAdvice: Array.isArray(data.personalized_advice) 
          ? data.personalized_advice 
          : [data.personalized_advice || "No financial advice available"]
      };
  
      console.log('Processed Result:', result); // Debugging
      setClassificationResult(result);
  
    } catch (err) {
      console.error("Full Error:", err); // Debugging
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  function AnalysisResults({ result }) {
    if (!result) return <div className="text-gray-500">No analysis results yet</div>;
  
    return (
      <div className="space-y-4">
        {/* Document Analysis Section */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-bold mb-2">Document Analysis</h2>
          <div className="space-y-1">
            <p><strong>Category:</strong> {result.documentAnalysis?.category || 'N/A'}</p>
            <p><strong>Match:</strong> {result.documentAnalysis?.match || 'N/A'}</p>
            <p><strong>Confidence:</strong> {result.documentAnalysis?.confidence || '0%'}</p>
          </div>
        </div>
  
        {/* Financial Advice Section */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-bold mb-2">Financial Advice</h3>
          {result.financialAdvice?.length > 0 ? (
            result.financialAdvice.map((advice, index) => (
              <div key={index} className="mb-3 p-3 bg-gray-50 rounded">
                {advice.split('\n').map((line, i) => (
                  <p key={i} className="mb-2 last:mb-0">{line}</p>
                ))}
              </div>
            ))
          ) : (
            <p className="text-gray-500">No financial advice available</p>
          )}
        </div>
  
        {/* Categorized Spending Section */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-bold mb-2">Spending Breakdown</h3>
          {Object.entries(result.categorizedSpending).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(result.categorizedSpending).map(([vendor, details]) => (
                <div key={vendor} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">
                    {vendor.replace(/[*:-]+/g, '').trim() || 'Unknown Vendor'}
                  </span>
                  <div className="text-right">
                    <p className="font-semibold">{details.amount || 'N/A'}</p>
                    <p className="text-sm text-gray-600">{details.category || 'Uncategorized'}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No spending categories identified</p>
          )}
        </div>
      </div>
    );
  }
  


  const fetchLoanApproval = async () => {
    setIsLoading(true);
    setError('');
    try {
      if (!customerId) return setError('Customer ID required');
      
      const data = await queuedRequest(`http://127.0.0.1:8000/loan/${customerId}`);
      setLoanApproval(data.loan_approval_insight);
    } catch (err) {
      handleApiError(err, 'Loan check');
    } finally {
      setIsLoading(false);
    }
  };

  
  const fetchFraudRisk = async () => {
    setFraudData({ isLoading: true, error: null, result: null });

    try {
        if (!customerId) {
            throw new Error('❌ Customer ID required');
        }

        console.log("🔄 Fetching fraud risk for:", customerId);

        const response = await fetch(`http://127.0.0.1:8000/fraud/${customerId}`, {
            method: "GET",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        });

        if (!response.ok) {
            throw new Error(`❌ API Error: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        console.log("✅ API Response:", data);

        // Update both states for compatibility
        setFraudRisk(data.fraud_advisory || "No fraud advisory available");
        setFraudData({
            isLoading: false,
            error: null,
            result: {
                isFraud: data.is_fraud,
                confidence: data.confidence_score,
                advisory: data.fraud_advisory,
                features: data.top_feature_importance
            }
        });

    } catch (err) {
        console.error("❌ Fraud Check Error:", err);
        setFraudData({
            isLoading: false,
            error: err.message,
            result: null
        });
        setFraudRisk("Error checking fraud risk");
    }
};

  <div>
    {fraudData.isLoading && <p>⏳ Checking fraud risk...</p>}
    
    {fraudData.error && <p style={{ color: 'red', fontWeight: 'bold' }}>❌ {fraudData.error}</p>}
    
    {fraudData.result && (
    <ResponseBox 
        title="Fraud Risk Analysis" 
        content={`Risk Level: ${fraudData.result.isFraud ? "⚠️ High Risk" : "✅ Low Risk"}
        Confidence: ${fraudData.result.confidence?.toFixed(2)}
        Advisory: ${fraudData.result.advisory}

        Key Risk Factors:
        ${Object.entries(fraudData.result.features || {})
            .map(([feature, value]) => `• ${feature}: ${value?.toFixed(2)}`)
            .join('\n')}`}
            />
        )}
</div>


  const fetchChurnRisk = async () => {
    setIsLoading(true);
    setError('');
    if (!customerId) return setError('❌ Customer ID required.');
    
  
    try {
      // ✅ Step 1: Check if the user has given consent
      const hasConsent = await checkUserConsent();
      if (!hasConsent) {
        return setError("❌ User has not given consent for AI predictions.");
      }
  
      // ✅ Step 2: Proceed with Churn Risk Prediction
      const data = await queuedRequest(`http://127.0.0.1:8000/churn/${customerId}`);
  
      // ✅ Step 3: Validate Response
      if (!data || typeof data !== 'object') {
        throw new Error("❌ Invalid response format from server.");
      }
      if (!("churn_risk" in data)) {
        throw new Error("⚠️ Server response missing churn risk data.");
      }
  
      // ✅ Step 4: Update state safely
      setChurnRisk(`Risk: ${data.churn_risk} (Confidence: ${data.confidence ?? 'N/A'})`);
    } catch (err) {
      console.error("❌ Churn Analysis Error:", err);
      handleApiError(err, 'Churn analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };
  
  
  const setUserConsent = async (consentGiven) => {
    setError('');
    if (!customerId) return setError('❌ Customer ID required.');
  
    try {
      const response = await queuedRequest(`http://127.0.0.1:8000/consent/${customerId}`, {
        method: "POST",
        data: JSON.stringify({ consent_given: consentGiven }),  // ✅ Correct JSON format
        headers: { 'Content-Type': 'application/json' }
      });
  
      // ✅ Ensure response is valid before using it
      if (!response || typeof response !== "object") {
        throw new Error("❌ Invalid response from server.");
      }
      if (!("message" in response)) {
        throw new Error("⚠️ Server response missing expected message.");
      }
  
      alert(response.message);  // ✅ Safe to access message
    } catch (err) {
      console.error("❌ Consent API Error:", err);
      handleApiError(err, 'Consent update failed.');
    }
  };
  
  const checkUserConsent = async () => {
    setError('');
    if (!customerId) return setError('❌ Customer ID required.');
  
    try {
      const response = await queuedRequest(`http://127.0.0.1:8000/consent/${customerId}`);
  
      if (!response || typeof response !== "object") {
        throw new Error("❌ Unexpected server response.");
      }
      if (!("consent_given" in response)) {
        throw new Error("⚠️ Server response missing consent status.");
      }
  
      const consentGiven = response.consent_given ?? false;
      alert(`User Consent: ${consentGiven ? "✅ Given" : "❌ Not Given"}`);
      return consentGiven;
    } catch (err) {
      console.error("❌ Check Consent API Error:", err);
      handleApiError(err, 'Consent check failed.');
      return false;
    }
  };
  

  const fetchWealthAdvice = async () => {
    setIsLoading(true);
    setError('');
    try {
      if (!customerId) return setError('Customer ID required');
      
      const data = await queuedRequest(`http://127.0.0.1:8000/wealth-management/${customerId}`);
      setWealthAdvice(data.wealth_management_advice);
    } catch (err) {
      handleApiError(err, 'Wealth advice');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSubscriptionAdvice = async () => {
    setIsLoading(true);
    setError('');
    try {
      if (!customerId) return setError('Customer ID required');
      
      const data = await queuedRequest(`http://127.0.0.1:8000/subscription-advice/${customerId}`);
      setSubscriptionAdvice(data.subscription_management_advice);
    } catch (err) {
      handleApiError(err, 'Subscription check');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchMarketInsights = async () => {
    setIsLoading(true);
    setError('');
    try {
      if (!businessId) return setError('Business ID required');
      
      const data = await queuedRequest(`http://127.0.0.1:8000/market-insights/${businessId}`);
      setMarketInsights(data.market_insight_advice);
    } catch (err) {
      handleApiError(err, 'Market insights');
    } finally {
      setIsLoading(false);
    }
  };

  const startVoiceInput = () => {
    if (!micEnabled) return; // ✅ Prevents listening if mic is disabled
    setError('');
    clearAll();

    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
        setError('Voice input not supported');
        return;
    }

    // ✅ Stop ongoing speech synthesis if the button is clicked again
    window.speechSynthesis.cancel();

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognitionRef.current = recognition;

    let speechTimeout = setTimeout(() => {
        recognition.stop();
        handleUserMessage(''); // ✅ Process output if no speech detected in 5s
    }, 5000);

    recognition.onresult = (event) => {
        clearTimeout(speechTimeout);
        const voiceText = event.results[0][0].transcript;
        handleUserMessage(voiceText);
    };

    recognition.onerror = (e) => {
        clearTimeout(speechTimeout);
        setError(`Voice error: ${e.error}`);
        setListening(false);
    };

    recognition.start();
};





// ✅ New Stop Function to Stop Both Mic & Speech
const stopListening = () => {
    if (recognitionRef.current) {
        recognitionRef.current.stop();
        setListening(false);
    }
    window.speechSynthesis.cancel();
};




  const [messages, setMessages] = useState([]);  // ✅ Add this to manage chatbot messages

  const handleUserMessage = async (message) => {
    setMessages((prev) => [...prev, { text: message, sender: "user" }]);
    const response = await fetchChatbotResponse(message);

    // ✅ Ensure only the actual text is spoken (not the whole object)
    const responseText = response.response ? response.response : "Sorry, I couldn't process your request.";

    setMessages((prev) => [...prev, { text: responseText, sender: "bot" }]);
    speakResponse(responseText); // ✅ Now it speaks only the chatbot's response text
};


const speakResponse = (text) => {
  if (!voiceEnabled) return; // ✅ Prevents speaking if voice output is disabled
  const utterance = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(utterance);
};

    
  
  return (
    <div style={{ 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f5f5f5',
      borderRadius: '10px'
    }}>
      <div style={{
        position: 'fixed',
        top: '10px',
        right: '10px',
        padding: '8px 15px',
        borderRadius: '20px',
        backgroundColor: networkStatus === 'online' ? '#4CAF50' : '#f44336',
        color: 'white',
        fontWeight: 'bold',
        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
        zIndex: 1000
      }}>
        {networkStatus.toUpperCase()} {networkStatus === 'online' ? '✓' : '⚠️'}
      </div>

      <div>
      <h1 style={{
    textAlign: 'center',
    fontSize: '1.5rem', // ✅ Increased size for better readability
    fontWeight: 'bold',
    background: 'linear-gradient(to right, #FF9933, #ffffff, #138808)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    color: '#222',
    textShadow: `
        2px 2px 4px rgba(0, 0, 0, 0.8), 
        4px 4px 8px rgba(0, 0, 0, 0.7)
    `,  // ✅ More layered shadows for depth
    padding: '15px 20px',  // ✅ Balanced spacing
    borderRadius: '10px',  
    display: 'inline-block',
    letterSpacing: '2px',  // ✅ Enhanced spacing for legibility
    fontFamily: '"Poppins", Arial Black, sans-serif', // ✅ More premium font
    textTransform: 'uppercase',
    border: '3px solid rgba(0, 0, 0, 0.7)', // ✅ Stronger border contrast
    boxShadow: '0px 6px 15px rgba(0, 0, 0, 0.6)', // ✅ More prominent shadow for elevation
}}>
    SynaPti-Q
</h1>


      </div>


      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <input
          placeholder="Customer ID"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          style={inputStyle}
          disabled={isLoading}
        />
        <input
          placeholder="Business ID"
          value={businessId}
          onChange={(e) => setBusinessId(e.target.value)}
          style={inputStyle}
          disabled={isLoading}
        />

        <div style={{ margin: '15px 0' }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setSelectedFile(e.target.files[0])}
            style={inputStyle}
            disabled={isLoading}
          />
          <button 
            style={{ ...buttonStyle, backgroundColor: '#9C27B0', width: '100%' }}
            onClick={fetchDocumentClassification}
            disabled={isLoading}
          >
            📄 Analyze Document
          </button>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center', marginTop: '15px' }}>
        <button style={buttonStyle} onClick={fetchFinancialAdvice} disabled={isLoading}>📊 Financial Advice</button>
        <button style={buttonStyle} onClick={fetchLoanApproval} disabled={isLoading}>💳 Loan Status</button>
        <button style={buttonStyle} onClick={fetchFraudRisk} disabled={isLoading}>⚠️ Fraud Risk</button>
        <button style={buttonStyle} onClick={fetchChurnRisk} disabled={isLoading}>📉 Churn Risk</button>
     

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center', marginTop: '10px' }}>
        <button style={{ ...buttonStyle, backgroundColor: '#4CAF50' }} onClick={() => setUserConsent(true)} disabled={isLoading}>✅ Give Consent</button>
        <button style={{ ...buttonStyle, backgroundColor: '#f44336' }} onClick={() => setUserConsent(false)} disabled={isLoading}>❌ Revoke Consent</button>
        <button style={{ ...buttonStyle, backgroundColor: '#FFC107' }} onClick={checkUserConsent} disabled={isLoading}>🔍 Check Consent</button>
      </div>
    </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
          <button style={buttonStyle} onClick={fetchWealthAdvice} disabled={isLoading}>📈 Wealth Management</button>
          <button style={buttonStyle} onClick={fetchSubscriptionAdvice} disabled={isLoading}>🔄 Subscription</button>
          <button style={buttonStyle} onClick={fetchMarketInsights} disabled={isLoading}>🌐 Market Insights</button>
        </div>
      </div>

      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
          <input
            placeholder="Ask a financial question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ ...inputStyle, flex: 1 }}
            disabled={isLoading}
          />
          <button style={{ ...buttonStyle, backgroundColor: '#4CAF50' }} onClick={startVoiceInput} disabled={isLoading}>
            🎙️ Voice
          </button>
          <button 
              style={{ ...buttonStyle, backgroundColor: '#2196F3' }} 
              onClick={() => fetchChatbotResponse(query, customerId || null).then(data => setChatbotResponse(data.response))}
              disabled={isLoading}
          >
              🔍 Ask
          </button>
          <button onClick={stopListening}>
              Stop Voice & Mic
          </button>




        </div>

        {error && (
          <div style={{ 
            padding: '10px', 
            backgroundColor: '#ffebee', 
            color: '#b71c1c',
            borderRadius: '5px',
            marginBottom: '15px'
          }}>
            {error}
            {networkStatus === 'offline' && (
              <div style={{ fontSize: '0.9em', marginTop: '5px' }}>
                Queued requests: {requestQueue.length}
              </div>
            )}
          </div>
        )}

        <div style={{ marginTop: '20px' }}>
          {classificationResult && (
            <ResponseBox
              title="Document Analysis"
              content={`Document Analysis:
        Category: ${classificationResult.documentAnalysis.category}
        Match: ${classificationResult.documentAnalysis.match}
        Confidence: ${classificationResult.documentAnalysis.confidence}

        Extracted Text:
        ${classificationResult.extractedText}

        Financial Insights:
        ${classificationResult.financialAdvice.map(advice => `• ${advice}`).join('\n')}

        Spending Categories:
        ${Object.entries(classificationResult.categorizedSpending)
          .map(([category, details]) => `• ${category}: ${details.amount} (${details.category})`)
          .join('\n')}`}
            />
          )}
          {chatbotResponse && <ResponseBox title="AI Response" content={chatbotResponse} />}
          {financialAdvice && <ResponseBox title="Financial Advice" content={financialAdvice} />}
          {loanApproval && <ResponseBox title="Loan Status" content={loanApproval} />}
          {fraudRisk && <ResponseBox title="Fraud Risk" content={fraudRisk} />}
          {churnRisk && <ResponseBox title="Churn Risk" content={churnRisk} />}
          {wealthAdvice && <ResponseBox title="Wealth Advice" content={wealthAdvice} />}
          {subscriptionAdvice && <ResponseBox title="Subscription Advice" content={subscriptionAdvice} />}
          {marketInsights && <ResponseBox title="Market Insights" content={marketInsights} />}
        </div>

        <button 
          onClick={clearAll}
          style={{
            marginTop: '25px',
            padding: '10px 25px',
            backgroundColor: '#e53935',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            width: '100%',
            fontSize: '16px'
          }}
          disabled={isLoading}
        >
          🗑️ Clear All
        </button>
      </div>
    </div>
  );
};

const ResponseBox = ({ title, content }) => (
  <div style={{ 
    padding: '15px', 
    backgroundColor: '#f8f9fa', 
    borderRadius: '6px', 
    marginBottom: '15px',
    borderLeft: `4px solid #2196F3`
  }}>
    <h4 style={{ margin: '0 0 10px 0', color: '#2c3e50' }}>{title}</h4>
    <p style={{ margin: 0, color: '#4a4a4a', whiteSpace: 'pre-line' }}>{content}</p>
  </div>
);

const inputStyle = {
  width: '100%',
  padding: '10px',
  marginBottom: '10px',
  border: '1px solid #ddd',
  borderRadius: '5px',
  fontSize: '16px'
};

const buttonStyle = {
  padding: '10px 15px',
  border: 'none',
  borderRadius: '5px',
  backgroundColor: '#2196F3',
  color: 'white',
  cursor: 'pointer',
  fontSize: '14px',
  flex: 1,
  transition: 'opacity 0.2s',
  ':disabled': {
    opacity: 0.6,
    cursor: 'not-allowed'
  }
};

export default App;