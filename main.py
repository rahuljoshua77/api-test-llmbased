from flask import Flask, request, jsonify, send_from_directory
import requests
import json
import re
import logging
from typing import List, Dict, Any
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/run-test', methods=['POST'])
def run_test():
    """
    Process API test requests.
    
    Expected JSON payload:
    {
        "curl_commands": string of one or more curl commands,
        "instructions": string describing the test,
        "api_key": Gemini API key,
        "settings": {
            "sequential": boolean,
            "validation": boolean,
            "extract_tokens": boolean
        }
    }
    """
    try:
        # Get request data
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        curl_commands = data.get('curl_commands', '')
        instructions = data.get('instructions', '')
        api_key = data.get('api_key', '')
        settings = data.get('settings', {})
        
        if not curl_commands:
            return jsonify({"error": "No curl commands provided"}), 400
        
        if not api_key:
            return jsonify({"error": "No API key provided"}), 400
        
        # Parse curl commands
        try:
            parsed_commands = CurlParser.parse_multiple_commands(curl_commands)
            if not parsed_commands:
                return jsonify({"error": "Could not parse any valid curl commands"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Process with Gemini
        try:
            gemini_client = GeminiClient(api_key)
            test_plan = gemini_client.process_instructions(
                parsed_commands, instructions, settings
            )
        except Exception as e:
            return jsonify({"error": f"Gemini API error: {str(e)}"}), 500
        
        # Execute test plan
        try:
            tester = APITester(parsed_commands)
            results = tester.execute_test_plan(test_plan.get('test_plan', {}))
        except Exception as e:
            return jsonify({"error": f"Test execution error: {str(e)}"}), 500
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


class CurlParser:
    """Parse curl commands into request parameters."""
    
    @staticmethod
    def parse_curl(curl_command: str) -> Dict[str, Any]:
        """
        Parse a curl command into request parameters.
        """
        try:
            # Remove line continuation characters and newlines
            curl_command = curl_command.replace("\\\n", " ").strip()
            
            # Initialize parameters
            method = "GET"  # Default method
            url = ""
            headers = {}
            data = None
            
            # Extract method
            method_match = re.search(r'-X\s+([A-Z]+)', curl_command)
            if method_match:
                method = method_match.group(1)
            
            # Extract URL
            url_match = re.search(r'curl\s+(?:-X\s+[A-Z]+\s+)?[\'"]?(https?://[^\s\'"]+)[\'"]?', curl_command)
            if url_match:
                url = url_match.group(1)
            else:
                raise ValueError("URL not found in curl command")
            
            # Extract headers
            header_matches = re.finditer(r'-H\s+[\'"]([^:]+):\s*([^\'"]+)[\'"]', curl_command)
            for match in header_matches:
                key, value = match.groups()
                headers[key.strip()] = value.strip()
            
            # Extract data
            data_match = re.search(r'-d\s+[\'"](.*?)[\'"]', curl_command, re.DOTALL)
            if data_match:
                data_str = data_match.group(1)
                try:
                    # Try to parse as JSON
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    # If not JSON, use as-is
                    data = data_str
            
            return {
                "method": method,
                "url": url,
                "headers": headers,
                "data": data
            }
        
        except Exception as e:
            logger.error(f"Error parsing curl command: {e}")
            raise ValueError(f"Failed to parse curl command: {str(e)}")
    
    @staticmethod
    def parse_multiple_commands(curl_commands: str) -> List[Dict[str, Any]]:
        """
        Parse multiple curl commands separated by blank lines.
        """
        commands = []
        for cmd in re.split(r'\n\s*\n', curl_commands):
            if cmd.strip():
                try:
                    parsed = CurlParser.parse_curl(cmd.strip())
                    commands.append(parsed)
                except ValueError as e:
                    logger.warning(f"Skipping invalid command: {e}")
        
        return commands


class GeminiClient:
    """Client for interacting with the Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def process_instructions(self, parsed_commands: List[Dict[str, Any]], 
                             instructions: str, 
                             settings: Dict[str, bool]) -> Dict[str, Any]:
        """
        Process user instructions with Gemini API to create a test plan.
        
        Args:
            parsed_commands: List of parsed curl commands
            instructions: Natural language instructions
            settings: User settings
            
        Returns:
            Test plan structure
        """
        try:
            # Format commands for prompt
            formatted_commands = []
            for i, cmd in enumerate(parsed_commands):
                formatted_cmd = f"Command {i+1}:\n"
                formatted_cmd += f"  Method: {cmd['method']}\n"
                formatted_cmd += f"  URL: {cmd['url']}\n"
                
                if cmd['headers']:
                    formatted_cmd += "  Headers:\n"
                    for k, v in cmd['headers'].items():
                        formatted_cmd += f"    {k}: {v}\n"
                
                if cmd['data']:
                    formatted_cmd += "  Data:\n"
                    if isinstance(cmd['data'], dict):
                        formatted_cmd += f"    {json.dumps(cmd['data'], indent=2)}\n"
                    else:
                        formatted_cmd += f"    {cmd['data']}\n"
                
                formatted_commands.append(formatted_cmd)
            
            commands_text = "\n".join(formatted_commands)
            
            # Create prompt for Gemini
            prompt = f"""
            You are an API testing assistant. I'll provide you with a list of API commands and natural language instructions.
            Your task is to create a detailed test plan with steps that can be executed in sequence.

            API COMMANDS:
            {commands_text}

            USER INSTRUCTIONS:
            {instructions if instructions else "Execute the commands in sequence and validate the responses."}

            SETTINGS:
            - Sequential flow: {settings.get('sequential', True)}
            - Validate responses: {settings.get('validation', True)}
            - Auto-extract tokens: {settings.get('extract_tokens', True)}

            Please create a structured test plan with the following information:
            1. List of steps to execute
            2. For each step:
               - Which command to use
               - Any modifications needed (like adding tokens from previous responses)
               - What to validate in the response
               - How to extract data for subsequent steps (if needed)

            Respond with a JSON object with the following structure:
            {{
              "test_plan": {{
                "steps": [
                  {{
                    "name": "Step name",
                    "command_index": 0,  // index in the provided commands list
                    "modifications": {{
                      "headers": {{}},  // headers to add/replace
                      "url_params": {{}},  // URL parameters to add
                      "data": {{}}  // data fields to add/replace
                    }},
                    "validation": [
                      {{
                        "type": "status_code",
                        "expected": 200
                      }},
                      {{
                        "type": "json_path",
                        "path": "$.field",
                        "expected": "value"
                      }}
                    ],
                    "extractions": [
                      {{
                        "name": "token",
                        "type": "json_path",
                        "path": "$.token"
                      }}
                    ]
                  }}
                ]
              }}
            }}
            """
            
            # Send request to Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse JSON from response
            try:
                # Extract JSON content from response
                response_text = response.text
                
                # Find JSON content - it might be wrapped in code blocks
                json_match = re.search(r'``````', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find any content in code blocks
                    code_match = re.search(r'``````', response_text, re.DOTALL)
                    if code_match:
                        json_str = code_match.group(1)
                    else:
                        # Just use the whole response
                        json_str = response_text
                
                # Try to clean the JSON string
                json_str = re.sub(r'^[^{]*', '', json_str)  # Remove anything before the first {
                json_str = re.sub(r'[^}]*$', '', json_str)  # Remove anything after the last }
                
                test_plan = json.loads(json_str)
                return test_plan
            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini response: {e}")
                logger.error(f"Response: {response.text}")
                
                # Fallback: create a simple sequential plan
                steps = []
                for i, cmd in enumerate(parsed_commands):
                    steps.append({
                        "name": f"Execute command {i+1}",
                        "command_index": i,
                        "modifications": {},
                        "validation": [{"type": "status_code", "expected": 200}],
                        "extractions": []
                    })
                
                return {"test_plan": {"steps": steps}}
        
        except Exception as e:
            logger.error(f"Error in Gemini processing: {e}")
            raise


class APITester:
    """Execute API tests based on test plans."""
    
    def __init__(self, parsed_commands: List[Dict[str, Any]]):
        """Initialize with parsed commands."""
        self.commands = parsed_commands
        self.session = requests.Session()
        self.extracted_data = {}
        self.logs = []
    
    def execute_test_plan(self, test_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test plan.
        
        Args:
            test_plan: Test plan from Gemini
        
        Returns:
            Test results
        """
        steps = test_plan.get('steps', [])
        results = []
        
        self.log("Starting test execution")
        self.log(f"Total steps: {len(steps)}")
        
        for i, step in enumerate(steps):
            self.log(f"\nExecuting step {i+1}: {step.get('name', f'Step {i+1}')}")
            
            try:
                # Get the base command
                command_index = step.get('command_index', 0)
                if command_index >= len(self.commands):
                    raise ValueError(f"Command index {command_index} out of range")
                
                base_command = self.commands[command_index].copy()
                
                # Apply modifications
                modified_command = self.apply_modifications(
                    base_command, 
                    step.get('modifications', {})
                )
                
                # Execute request
                response = self.execute_request(modified_command)
                
                # Validate response
                validation_results = self.validate_response(
                    response, 
                    step.get('validation', [])
                )
                
                # Extract data
                self.extract_data(
                    response, 
                    step.get('extractions', [])
                )
                
                # Format request and response for display
                request_info = self.format_request_info(modified_command)
                response_info = self.format_response_info(response)
                
                # Add to results
                success = all(result.get('success', False) for result in validation_results)
                result_step = {
                    "name": step.get('name', f"Step {i+1}"),
                    "success": success,
                    "request": request_info,
                    "response": response_info,
                    "validations": validation_results,
                    "notes": "Extracted data: " + json.dumps(self.extracted_data) if self.extracted_data else ""
                }
                
                results.append(result_step)
                
                # If step failed and we're in sequential mode, stop execution
                if not success and test_plan.get('sequential', True):
                    self.log(f"Step {i+1} failed, stopping execution.")
                    break
            
            except Exception as e:
                self.log(f"Error executing step {i+1}: {str(e)}")
                results.append({
                    "name": step.get('name', f"Step {i+1}"),
                    "success": False,
                    "request": "Error: " + str(e),
                    "response": "N/A",
                    "notes": "Failed to execute step"
                })
                break
        
        return {
            "steps": results,
            "logs": self.logs
        }
    
    def apply_modifications(self, command: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to a command."""
        result = command.copy()
        
        # Modify headers
        if 'headers' in modifications:
            headers = result.get('headers', {}).copy()
            for k, v in modifications['headers'].items():
                # Replace variables
                if isinstance(v, str):
                    v = self.replace_variables(v)
                headers[k] = v
            result['headers'] = headers
        
        # Modify URL parameters
        if 'url_params' in modifications and modifications['url_params']:
            url = result['url']
            params = {}
            
            # Extract existing params
            if '?' in url:
                base_url, query = url.split('?', 1)
                for param in query.split('&'):
                    if '=' in param:
                        k, v = param.split('=', 1)
                        params[k] = v
            else:
                base_url = url
            
            # Add new params
            for k, v in modifications['url_params'].items():
                # Replace variables
                if isinstance(v, str):
                    v = self.replace_variables(v)
                params[k] = v
            
            # Rebuild URL
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                result['url'] = f"{base_url}?{'&'.join(param_strs)}"
        
        # Modify data
        if 'data' in modifications and modifications['data']:
            data = result.get('data', {})
            if isinstance(data, dict) and isinstance(modifications['data'], dict):
                for k, v in modifications['data'].items():
                    # Replace variables
                    if isinstance(v, str):
                        v = self.replace_variables(v)
                    data[k] = v
                result['data'] = data
            elif isinstance(modifications['data'], dict):
                # Convert string data to dict if modifications are a dict
                try:
                    data_dict = json.loads(data) if isinstance(data, str) else {}
                    for k, v in modifications['data'].items():
                        # Replace variables
                        if isinstance(v, str):
                            v = self.replace_variables(v)
                        data_dict[k] = v
                    result['data'] = data_dict
                except:
                    self.log("Warning: Could not apply data modifications")
            else:
                # Replace the entire data
                result['data'] = self.replace_variables(modifications['data']) if isinstance(modifications['data'], str) else modifications['data']
        
        return result
    
    def replace_variables(self, text: str) -> str:
        """Replace variables in text with extracted values."""
        if not isinstance(text, str):
            return text
        
        # Replace ${variable} with extracted data
        for var_name, value in self.extracted_data.items():
            if isinstance(value, str):
                text = text.replace(f"${{{var_name}}}", value)
        
        return text
    
    def execute_request(self, command: Dict[str, Any]) -> requests.Response:
        """Execute an HTTP request."""
        method = command['method'].upper()
        url = command['url']
        headers = command.get('headers', {})
        data = command.get('data')
        
        data_str = json.dumps(data) if isinstance(data, dict) else data
        self.log(f"Sending {method} request to {url}")
        if headers:
            self.log(f"Headers: {json.dumps(headers)}")
        if data:
            self.log(f"Data: {data_str}")
        
        # Convert dict data to JSON string with proper content type
        if isinstance(data, dict):
            if 'Content-Type' not in headers:
                headers['Content-Type'] = 'application/json'
            data = json.dumps(data)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=data
            )
            
            self.log(f"Received response: HTTP {response.status_code}")
            try:
                response_json = response.json()
                self.log(f"Response body: {json.dumps(response_json, indent=2)}")
            except:
                self.log(f"Response body: {response.text[:200]}...")
            
            return response
        
        except requests.RequestException as e:
            self.log(f"Request failed: {str(e)}")
            raise
    
    def validate_response(self, response: requests.Response, validations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a response against validation rules."""
        results = []
        
        for validation in validations:
            validation_type = validation.get('type', '')
            expected = validation.get('expected')
            actual = None
            success = False
            message = ""
            
            try:
                if validation_type == 'status_code':
                    actual = response.status_code
                    success = actual == expected
                    message = f"Expected status {expected}, got {actual}"
                
                elif validation_type == 'json_path':
                    path = validation.get('path', '')
                    try:
                        # Simple JSON path implementation
                        json_data = response.json()
                        
                        # Handle path like $.field.subfield
                        if path.startswith('$.'):
                            path = path[2:]
                        
                        parts = path.split('.')
                        value = json_data
                        for part in parts:
                            if part in value:
                                value = value[part]
                            else:
                                raise ValueError(f"Path {path} not found in response")
                        
                        actual = value
                        
                        # Compare values
                        if expected is not None:
                            if isinstance(expected, str) and expected.startswith('regex:'):
                                # Regex comparison
                                regex = expected[6:]
                                success = bool(re.match(regex, str(actual)))
                                message = f"Expected to match {regex}, got {actual}"
                            else:
                                # Direct comparison
                                success = actual == expected
                                message = f"Expected {expected}, got {actual}"
                        else:
                            # Just check existence
                            success = True
                            message = f"Found value: {actual}"
                    
                    except json.JSONDecodeError:
                        actual = "Not JSON"
                        success = False
                        message = "Response is not valid JSON"
                    
                    except ValueError as e:
                        actual = "Not found"
                        success = False
                        message = str(e)
                
                elif validation_type == 'contains':
                    text = validation.get('text', '')
                    response_body = response.text
                    success = text in response_body
                    actual = response_body[:50] + "..." if len(response_body) > 50 else response_body
                    message = f"Expected to find '{text}' in response"
                
                elif validation_type == 'header':
                    header_name = validation.get('name', '')
                    header_value = validation.get('value', '')
                    
                    if header_name in response.headers:
                        actual = response.headers[header_name]
                        success = header_value in actual
                        message = f"Expected header {header_name} to be {header_value}, got {actual}"
                    else:
                        actual = None
                        success = False
                        message = f"Header {header_name} not found in response"
            
            except Exception as e:
                success = False
                message = f"Validation error: {str(e)}"
            
            results.append({
                "type": validation_type,
                "expected": expected,
                "actual": actual,
                "success": success,
                "message": message
            })
            
            self.log(f"Validation ({validation_type}): {message} - {'✅ Passed' if success else '❌ Failed'}")
        
        return results
    
    def extract_data(self, response: requests.Response, extractions: List[Dict[str, Any]]) -> None:
        """Extract data from response for use in subsequent steps."""
        for extraction in extractions:
            extraction_name = extraction.get('name', '')
            extraction_type = extraction.get('type', '')
            
            try:
                if extraction_type == 'json_path':
                    path = extraction.get('path', '')
                    
                    try:
                        # Simple JSON path implementation
                        json_data = response.json()
                        
                        # Handle path like $.field.subfield
                        if path.startswith('$.'):
                            path = path[2:]
                        
                        parts = path.split('.')
                        value = json_data
                        for part in parts:
                            if part in value:
                                value = value[part]
                            else:
                                raise ValueError(f"Path {path} not found in response")
                        
                        # Store extracted value
                        self.extracted_data[extraction_name] = value
                        self.log(f"Extracted {extraction_name} = {value}")
                    
                    except json.JSONDecodeError:
                        self.log(f"Failed to extract {extraction_name}: Response is not valid JSON")
                    
                    except ValueError as e:
                        self.log(f"Failed to extract {extraction_name}: {str(e)}")
                
                elif extraction_type == 'regex':
                    pattern = extraction.get('pattern', '')
                    text = response.text
                    
                    match = re.search(pattern, text)
                    if match:
                        value = match.group(1) if match.groups() else match.group(0)
                        self.extracted_data[extraction_name] = value
                        self.log(f"Extracted {extraction_name} = {value}")
                    else:
                        self.log(f"Failed to extract {extraction_name}: Pattern {pattern} not found")
                
                elif extraction_type == 'header':
                    header_name = extraction.get('header', '')
                    
                    if header_name in response.headers:
                        value = response.headers[header_name]
                        self.extracted_data[extraction_name] = value
                        self.log(f"Extracted {extraction_name} = {value} from header {header_name}")
                    else:
                        self.log(f"Failed to extract {extraction_name}: Header {header_name} not found")
            
            except Exception as e:
                self.log(f"Error extracting {extraction_name}: {str(e)}")
    
    def format_request_info(self, command: Dict[str, Any]) -> str:
        """Format request information for display."""
        method = command['method'].upper()
        url = command['url']
        headers = command.get('headers', {})
        data = command.get('data')
        
        info = f"{method} {url}"
        
        if headers:
            headers_str = ", ".join(f"{k}: {v}" for k, v in headers.items())
            info += f"\nHeaders: {headers_str}"
        
        if data:
            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            
            # Truncate long data
            if len(data_str) > 200:
                data_str = data_str[:197] + "..."
            
            info += f"\nData: {data_str}"
        
        return info
    
    def format_response_info(self, response: requests.Response) -> str:
        """Format response information for display."""
        info = f"Status: {response.status_code}"
        
        try:
            data = response.json()
            data_str = json.dumps(data, indent=2)
            
            # Truncate long responses
            if len(data_str) > 300:
                data_str = data_str[:297] + "..."
            
            info += f"\nBody: {data_str}"
        except:
            text = response.text
            if len(text) > 300:
                text = text[:297] + "..."
            info += f"\nBody: {text}"
        
        return info
    
    def log(self, message: str) -> None:
        """Add a message to the logs."""
        self.logs.append(message)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
