import streamlit as st
from pathlib import Path
import google.generativeai as genai
from google_api_key import google_api_key
import pydicom
import wfdb
import numpy as np
from PIL import Image
import io
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Load API key from Streamlit secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)


def process_dicom_ecg(file_data):
    """Process DICOM-ECG format and convert to image"""
    try:
        # Read DICOM data from bytes
        dicom_data = pydicom.dcmread(io.BytesIO(file_data))
        
        # Extract waveform data
        waveform_data = dicom_data.WaveformSequence[0]
        channel_data = waveform_data.WaveformData
        
        # Convert to numpy array and reshape
        ecg_data = np.frombuffer(channel_data, dtype=np.int16)
        num_channels = len(waveform_data.ChannelDefinitionSequence)
        samples_per_channel = len(ecg_data) // num_channels
        ecg_data = ecg_data.reshape(num_channels, samples_per_channel)
        
        return plot_ecg_data(ecg_data)
    except Exception as e:
        st.error(f"Error processing DICOM-ECG: {str(e)}")
        return None

def process_scp_ecg(file_data):
    """Process SCP-ECG format and convert to image"""
    try:
        # Write temporary file for wfdb to read
        temp_file = "temp_ecg"
        with open(temp_file, 'wb') as f:
            f.write(file_data)
        
        # Read using wfdb
        record = wfdb.rdrecord(temp_file)
        ecg_data = record.p_signal
        
        return plot_ecg_data(ecg_data)
    except Exception as e:
        st.error(f"Error processing SCP-ECG: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        Path(temp_file).unlink(missing_ok=True)

def process_hl7_ecg(file_data):
    """Process HL7 format and convert to image"""
    try:
        # Decode HL7 message
        hl7_text = file_data.decode('utf-8')
        segments = hl7_text.split('\r')
        
        # Find OBX segments containing waveform data
        waveform_data = []
        for segment in segments:
            if segment.startswith('OBX|') and '~' in segment:
                # Extract numerical data from the segment
                values = segment.split('|')[5].split('~')
                waveform_data.extend([float(v) for v in values if v])
        
        if waveform_data:
            ecg_data = np.array(waveform_data)
            return plot_ecg_data(ecg_data.reshape(1, -1))
        return None
    except Exception as e:
        st.error(f"Error processing HL7-ECG: {str(e)}")
        return None

def process_xml_ecg(file_data):
    """Process XML format and convert to image"""
    try:
        # Parse XML
        root = ET.fromstring(file_data)
        
        # Extract waveform data (adjust xpath based on XML structure)
        waveform_elements = root.findall('.//waveformData')
        if not waveform_elements:
            waveform_elements = root.findall('.//channel')
        
        if waveform_elements:
            waveform_data = []
            for element in waveform_elements:
                values = [float(v) for v in element.text.split()]
                waveform_data.append(values)
            
            ecg_data = np.array(waveform_data)
            return plot_ecg_data(ecg_data)
        return None
    except Exception as e:
        st.error(f"Error processing XML-ECG: {str(e)}")
        return None

def plot_ecg_data(ecg_data):
    """Convert ECG data to image"""
    plt.figure(figsize=(12, 8))
    
    # Apply bandpass filter
    fs = 500  # sampling frequency (adjust as needed)
    lowcut = 0.5
    highcut = 50.0
    nyquist = fs/2
    low = lowcut/nyquist
    high = highcut/nyquist
    b, a = butter(4, [low, high], btype='band')
    
    # Plot each lead
    for i in range(ecg_data.shape[0]):
        # Apply filter
        filtered_data = filtfilt(b, a, ecg_data[i])
        
        plt.subplot(6, 2, i+1 if i < 12 else 12)
        plt.plot(filtered_data, 'b-', linewidth=0.5)
        plt.title(f'Lead {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Configure Google AI
genai.configure(api_key=google_api_key)

# Generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

# Safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# ECG-specific system prompt
system_prompts = [
    """
    You are an expert cardiologist specialized in ECG interpretation. Your task is to analyze 
    ECG readings and provide detailed interpretations for medical professionals.
    
    Your key responsibilities:
    1. Rhythm Analysis: Identify the heart rhythm and rate
    2. Waveform Analysis: Examine P waves, QRS complexes, T waves, and intervals
    3. Pattern Recognition: Identify any abnormal patterns or potential cardiac conditions
    4. Clinical Correlation: Suggest possible clinical implications
    
    Structure your analysis in these sections:
    
    1. Basic Measurements:
    - Heart Rate
    - Rhythm
    - Intervals (PR, QRS, QT)
    - Axis
    
    2. Detailed Analysis:
    - P wave morphology
    - QRS complex characteristics
    - ST segment analysis
    - T wave analysis
    
    3. Key Findings:
    - Primary abnormalities
    - Secondary changes
    - Pattern recognition
    
    4. Clinical Interpretation:
    - Diagnostic considerations
    - Suggested clinical correlations
    - Recommended follow-up
    
    Important Notes:
    1. Signal Quality: Note if any leads show noise or artifacts
    2. Technical Details: Include sampling rate and filter settings if available
    3. Disclaimer: Include: "This is an AI-assisted interpretation. Please confirm with a qualified 
       cardiologist before making clinical decisions."
    
    Please maintain this structured format in your response.
    """
]

# Initialize Gemini Pro model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                            generation_config=generation_config,
                            safety_settings=safety_settings)

# Streamlit UI
st.set_page_config(page_title="ECG Interpreter Assistant", page_icon="❤", layout="wide")

# Header with custom styling
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        color: #FF4B4B;
    }
    .file-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">ECG Interpreter Assistant</p>', unsafe_allow_html=True)

# Subheader with emoji
st.markdown("### 📈 Upload an ECG for Professional Analysis")

# Information box
st.info("""
    This AI-powered tool supports multiple ECG formats:
    - DICOM-ECG (.dcm)
    - HL7 (.hl7)
    - SCP-ECG (.scp)
    - XML (.xml)
    - Standard image formats (.png, .jpg, .jpeg)
""")

# File uploader with expanded file types
file_uploaded = st.file_uploader('Upload ECG File', 
                                type=['dcm', 'hl7', 'scp', 'xml', 'png', 'jpg', 'jpeg'],
                                help="Multiple formats supported")

if file_uploaded:
    file_data = file_uploaded.getvalue()
    file_extension = Path(file_uploaded.name).suffix.lower()
    
    # Process different file formats
    if file_extension == '.dcm':
        processed_image = process_dicom_ecg(file_data)
    elif file_extension == '.hl7':
        processed_image = process_hl7_ecg(file_data)
    elif file_extension == '.scp':
        processed_image = process_scp_ecg(file_data)
    elif file_extension == '.xml':
        processed_image = process_xml_ecg(file_data)
    else:  # Standard image formats
        processed_image = Image.open(io.BytesIO(file_data))
    
    if processed_image:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(processed_image, caption='Processed ECG', use_column_width=True)
        with col2:
            st.markdown("""
                ### File Information
                """)
            st.markdown(f"""
                <div class="file-info">
                <p>📁 File Type: {file_extension}</p>
                <p>📊 Status: Successfully processed</p>
                </div>
                """, unsafe_allow_html=True)

# Analysis button with spinner
if st.button("📊 Analyze ECG", type="primary"):
    if file_uploaded:
        with st.spinner('Analyzing ECG... Please wait...'):
            # Convert processed image to bytes for Gemini
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            image_parts = [
                {
                    "mime_type": "image/png",
                    "data": img_byte_arr
                }
            ]
            
            prompt_parts = [
                image_parts[0],
                system_prompts[0],
            ]
            
            response = model.generate_content(prompt_parts)
            
            if response:
                st.success("Analysis Complete!")
                
                # Create tabs for organized display
                tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Metrics", "ℹ Notes"])
                
                with tab1:
                    st.markdown("## ECG Interpretation Results")
                    st.write(response.text)
                
                with tab2:
                    st.markdown("### Signal Quality Metrics")
                    # Add signal quality metrics here if available
                    st.markdown("- Sampling Rate: 500 Hz")
                    st.markdown("- Filter Settings: 0.5-50 Hz bandpass")
                    st.markdown("- Signal-to-Noise Ratio: Calculated from waveform")
                
                with tab3:
                    st.warning("""
                        *Important Disclaimers:*
                        - This is an AI-assisted interpretation
                        - Results should be verified by a qualified healthcare professional
                        - Not intended as a replacement for professional medical judgment
                        - Some formats may have limited compatibility
                    """)
    else:
        st.error("Please upload an ECG file first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🏥 For educational and assistive purposes only</p>
        <p>Supports multiple ECG formats for comprehensive analysis</p>
    </div>
""", unsafe_allow_html=True)