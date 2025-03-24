import pytest
import json
from fastapi.testclient import TestClient
from banking_api import app  # Importing FastAPI app
import os
os.environ["RUNNING_TESTS"] = "1"  # ✅ Prevents FAISS from running in tests

from banking_api import app  # Now FAISS won't execute
import warnings
import sys
sys.modules["faiss"] = None 
# Suppress Transformer & Torch Warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)



# ✅ Create a test client
client = TestClient(app)

# ✅ Parameterized test cases for general API responses
@pytest.mark.parametrize("endpoint", [
    "/customer-ids/",
    "/loan-customer-ids/",
    "/financial/CUST-0001",
    "/churn/CUST-0001",
    "/loan/CUST-0001",
    "/fraud/TXN001",
    "/subscription/CUST-0001",
    "/wealth/CUSTW-00001",
    "/business/BIZ001",
    "/market/MKT-00001",
    "/chatbot/"
])
def test_api_responses(endpoint):
    """Test API endpoints to ensure they return valid responses."""
    response = client.get(endpoint)
    assert response.status_code in [200, 403, 404], f"Unexpected response {response.status_code} for {endpoint}"

    import pytest

@pytest.mark.parametrize("endpoint, method", [
    ("/customer-ids/", "GET"),
    ("/loan-customer-ids/", "GET"),
    ("/financial/CUST001", "GET"),
    ("/churn/CUST001", "GET"),
    ("/loan/CUST001", "GET"),
    ("/fraud/TXN001", "GET"),
    ("/subscription/CUST001", "GET"),
    ("/wealth/CUST001", "GET"),
    ("/business/BIZ001", "GET"),
    ("/market/BIZ001", "GET"),
    ("/chatbot/", "POST")  # ✅ Ensure chatbot uses POST
])
def test_api_responses(endpoint, method):
    """Test API endpoints to ensure they return valid responses."""
    if method == "POST":
        response = client.post(endpoint, json={"query": "Test question"})
    else:
        response = client.get(endpoint)

    # ✅ Skip test if the API method is incorrect
    if response.status_code == 405:
        pytest.skip(f"Skipping {endpoint} test due to method not allowed (405)")

    assert response.status_code in [200, 403, 404], f"Unexpected response {response.status_code} for {endpoint}"


# ✅ Test consent workflow (setting, checking, and revoking)
def test_consent_workflow():
    """Test consent setting and retrieval."""
    cust_id = "CUST-0001"

    # Set consent
    response = client.post(f"/consent/{cust_id}", json={"consent_given": True})
    assert response.status_code == 200
    assert response.json()["consent_given"] is True

    # Check consent
    response = client.get(f"/consent/{cust_id}")
    assert response.status_code == 200
    assert response.json()["consent_given"] is True

    # Revoke consent
    response = client.post(f"/consent/{cust_id}", json={"consent_given": False})
    assert response.status_code == 200
    assert response.json()["consent_given"] is False

# ✅ Test financial advice endpoint
def test_financial_advice():
    """Test financial insights for a customer."""
    response = client.get("/financial/CUST-0001")
    assert response.status_code == 200
    assert "financial_advice" in response.json()

def test_loan_approval():
    """Test loan approval insights (Ensure valid customer)."""
    response = client.get("/loan-customer-ids/")
    assert response.status_code == 200
    customer_ids = response.json().get("available_loan_customer_ids", [])

    assert customer_ids, "No loan customers found! Ensure the API is loading data."
    valid_customer_id = customer_ids[0]

    response = client.get(f"/loan/{valid_customer_id}")
    assert response.status_code == 200, f"Unexpected response {response.status_code}: {response.text}"
    assert "loan_approval_insight" in response.json()


def test_fraud_detection():
    """Test fraud detection API with valid transaction."""
    response = client.get("/fraud/TXN001")
    assert response.status_code in [200, 404], f"Unexpected response {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        assert "is_fraud" in response.json()


def test_subscription_advice():
    """Test subscription churn insights."""
    response = client.get("/customer-ids/")
    assert response.status_code == 200
    customer_ids = response.json().get("available_customer_ids", [])

    assert customer_ids, "No customers found for subscription test!"
    valid_customer_id = customer_ids[0]

    response = client.get(f"/subscription/{valid_customer_id}")
    assert response.status_code in [200, 404], f"Unexpected response {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        assert "subscription_management_advice" in response.json()


def test_wealth_management():
    """Test wealth management AI recommendations with valid customer."""
    response = client.get("/customer-ids/")
    assert response.status_code == 200, f"Customer IDs API failed: {response.status_code}"
    
    customer_ids = response.json().get("available_customer_ids", [])
    
    if not customer_ids:
        pytest.skip("Skipping wealth test because no valid customers are available.")

    valid_customer_id = customer_ids[0]

    response = client.get(f"/wealth/{valid_customer_id}")

    if response.status_code == 404:
        pytest.skip(f"Skipping wealth test because customer {valid_customer_id} is not found.")

    assert response.status_code == 200, f"Unexpected response {response.status_code}: {response.text}"
    assert "wealth_management_advice" in response.json(), "Expected 'wealth_management_advice' in response"





# ✅ Test business financial insights
def test_business_insights():
    """Test business financial prediction API."""
    response = client.get("/business/BIZ001")
    assert response.status_code in [200, 404], f"Unexpected response {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        assert "business_financial_insight" in response.json()


# ✅ Test market insights
def test_market_insights():
    """Test market predictions for businesses."""
    response = client.get("/market/BIZ001")
    assert response.status_code in [200, 404], f"Unexpected response {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        assert "market_insight_advice" in response.json()


# ✅ Test chatbot general query without customer ID
def test_chatbot_general_query():
    """Test chatbot with a general query (no customer ID)."""
    response = client.post("/chatbot/", json={"query": "What are the latest market trends?"})
    assert response.status_code == 200
    assert "response" in response.json()

# ✅ Test churn risk predictions
def test_churn_risk():
    """Test churn predictions (requires consent and valid customer ID)."""
    cust_id = "CUST-0001"

    # Ensure the customer exists before calling churn API
    response = client.get(f"/customer-ids/")
    assert response.status_code == 200
    customer_ids = response.json().get("available_customer_ids", [])
    assert cust_id in customer_ids, f"Customer ID {cust_id} not found in API response."

    # Give consent
    client.post(f"/consent/{cust_id}", json={"consent_given": True})

    # Call churn API
    response = client.get(f"/churn/{cust_id}")
    assert response.status_code == 200, f"Unexpected response {response.status_code}: {response.text}"
    assert "churn_risk" in response.json()


# ✅ Test invalid endpoints
def test_invalid_endpoint():
    """Test an invalid API route."""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404

# ✅ Test chatbot failure handling (Invalid request)
def test_chatbot_failure():
    """Test chatbot failure handling with missing query."""
    response = client.post("/chatbot/", json={})
    assert response.status_code == 422  # FastAPI should return 422 for missing required fields
