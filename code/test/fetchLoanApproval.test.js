import axios from "axios";
import { renderHook, act } from "@testing-library/react";
import { useState } from "react";

// Mock Axios
jest.mock("axios");

// ✅ Component Hook to Test
const useLoanApproval = () => {
  const [loanApproval, setLoanApproval] = useState("");
  const [error, setError] = useState("");

  const fetchLoanApproval = async (customerId) => {
    setError("");
    setLoanApproval("");

    if (!customerId.trim()) {
      setError("Please enter a valid Customer ID.");
      return;
    }

    try {
      const res = await axios.get(`http://127.0.0.1:8000/loan/${customerId}`);

      if (res.data && ("loan_approval_status" in res.data || "loan_approval_insight" in res.data)) {
        const approvalStatus = res.data.loan_approval_status ?? res.data.loan_approval_insight;
        setLoanApproval(approvalStatus ? `✅ Loan Approved: ${approvalStatus}` : "❌ Loan Denied");
      } else {
        setError("Unexpected API response format.");
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Could not fetch loan status");
    }
  };

  return { fetchLoanApproval, loanApproval, error };
};

// ✅ Unit Test Cases
describe("fetchLoanApproval()", () => {
  it("should return loan approval status when API responds successfully", async () => {
    // Mock successful API response
    axios.get.mockResolvedValue({
      data: { loan_approval_status: "Approved" },
    });

    // Use React hook for state
    const { result } = renderHook(() => useLoanApproval());

    // Call function inside `act()` to trigger state updates
    await act(async () => {
      await result.current.fetchLoanApproval("CUST-1234");
    });

    // Expect loan approval to be set
    expect(result.current.loanApproval).toBe("✅ Loan Approved: Approved");
    expect(result.current.error).toBe("");
  });

  it("should return error if Customer ID is empty", async () => {
    const { result } = renderHook(() => useLoanApproval());

    await act(async () => {
      await result.current.fetchLoanApproval("");
    });

    expect(result.current.error).toBe("Please enter a valid Customer ID.");
  });

  it("should return API error message on failure", async () => {
    // Mock API error response
    axios.get.mockRejectedValue({
      response: { data: { detail: "Customer not found" } },
    });

    const { result } = renderHook(() => useLoanApproval());

    await act(async () => {
      await result.current.fetchLoanApproval("INVALID-ID");
    });

    expect(result.current.error).toBe("Customer not found");
  });

  it("should handle unexpected API response", async () => {
    // Mock response with missing data
    axios.get.mockResolvedValue({
      data: {},
    });

    const { result } = renderHook(() => useLoanApproval());

    await act(async () => {
      await result.current.fetchLoanApproval("CUST-5678");
    });

    expect(result.current.error).toBe("Unexpected API response format.");
  });
});
