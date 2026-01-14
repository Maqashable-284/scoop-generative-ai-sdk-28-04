#!/bin/bash
# Test script to check AI prompt compliance
# Tests [TIP] and [QUICK_REPLIES] tag generation

BACKEND_URL="http://localhost:8080"
USER_ID="test_compliance_$(date +%s)"

echo "🧪 Testing Scoop AI Prompt Compliance"
echo "====================================="
echo "User ID: $USER_ID"
echo ""

# Test 1: Product recommendation (should have [TIP] and [QUICK_REPLIES])
echo "Test 1: Product Recommendation"
echo "-------------------------------"
echo "Query: 'მაჩვენე whey პროტეინები'"
echo ""

RESPONSE_1=$(curl -s -X POST "$BACKEND_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"მაჩვენე whey პროტეინები\"}")

echo "$RESPONSE_1" | jq -r '.response_text_geo' > /tmp/test1_response.txt

echo "Response saved to: /tmp/test1_response.txt"
echo ""

# Check for tags
if grep -q "\[TIP\]" /tmp/test1_response.txt; then
  echo "✅ [TIP] tag found"
else
  echo "❌ [TIP] tag MISSING"
fi

if grep -q "\[QUICK_REPLIES\]" /tmp/test1_response.txt; then
  echo "✅ [QUICK_REPLIES] tag found"
else
  echo "❌ [QUICK_REPLIES] tag MISSING"
fi

echo ""
echo "Quick Replies from API:"
echo "$RESPONSE_1" | jq -r '.quick_replies[] | "  - \(.title)"'

echo ""
echo "=========================================="
echo ""

# Test 2: Educational question (should have [TIP] and [QUICK_REPLIES])
echo "Test 2: Educational Question"
echo "----------------------------"
echo "Query: 'როგორ მივიღო კრეატინი?'"
echo ""

RESPONSE_2=$(curl -s -X POST "$BACKEND_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"message\": \"როგორ მივიღო კრეატინი?\"}")

echo "$RESPONSE_2" | jq -r '.response_text_geo' > /tmp/test2_response.txt

echo "Response saved to: /tmp/test2_response.txt"
echo ""

# Check for tags
if grep -q "\[TIP\]" /tmp/test2_response.txt; then
  echo "✅ [TIP] tag found"
else
  echo "❌ [TIP] tag MISSING"
fi

if grep -q "\[QUICK_REPLIES\]" /tmp/test2_response.txt; then
  echo "✅ [QUICK_REPLIES] tag found"
else
  echo "❌ [QUICK_REPLIES] tag MISSING"
fi

echo ""
echo "Quick Replies from API:"
echo "$RESPONSE_2" | jq -r '.quick_replies[] | "  - \(.title)"'

echo ""
echo "=========================================="
echo ""

# Display full responses
echo "Full Response 1:"
cat /tmp/test1_response.txt
echo ""
echo ""
echo "Full Response 2:"
cat /tmp/test2_response.txt
echo ""
