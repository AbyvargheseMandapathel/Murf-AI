from fastapi import UploadFile
import assemblyai as aai
from dotenv import load_dotenv
import os
import asyncio
import logging
from typing import Type
from google import genai
import time
from datetime import datetime, timedelta
from collections import deque
from services.murf_service import MurfService
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

load_dotenv()
aai.settings.api_key =  os.getenv('ASSEMBLYAI_API_KEY')

class AssemblyAIStreamingClient:
    def __init__(self, websocket, loop, sample_rate=16000, silence_threshold=1.5):
        self.websocket = websocket
        self.loop = loop
        self.silence_threshold = silence_threshold #second of silence to trigger LLM
        self.last_audio_time = None
        self.transcript_buffer = []
        self.llm_task = None
        self.is_processing = False
        
        
        #Initialize MurfService
        self.murf_service = MurfService(
            websocket = websocket,
            api_key=os.getenv("MURF_API_KEY")
        )
        
          # Initialize AssemblyAI client
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key= aai.settings.api_key,
                api_host= "streaming.assemblyai.com"
                
            )
        )
        # Set up event handlers
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)
        
        # Connect with parameters
        self.client.connect(StreamingParameters(
            sample_rate=sample_rate, format_turns=False
        ))
    
    def on_begin(self, client, event: BeginEvent):
       print(f"Session started: {event.id}")
       self.last_audio_time = time.time()
    
    def on_turn(self, client, event: TurnEvent):
        current_time = time.time()
        
        #update last audio time 
        self.last_audio_time = current_time
        
        
        # Print transcripts as they arrive
        print(f"{event.transcript} (end_of_turn={event.end_of_turn})")
       
       #Buffer the transcript
        if event.transcript.strip():
           self.transcript_buffer.append({
               'text': event.transcript,
                'timestamp': current_time,
                'end_of_turn': event.end_of_turn
           })
              
        # Check for silence and process if needed
        self.check_silence_and_process()  
      
      
        if event.end_of_turn and not event.turn_is_formatted:
          client.set_params(StreamingSessionParameters(format_turns=True))
    
    
    def check_silence_and_process(self):
        """Check if enough silence has passed and process transcript"""
        if (self.last_audio_time and 
            time.time() - self.last_audio_time > self.silence_threshold and
            self.transcript_buffer and 
            not self.is_processing):
            
            self.process_buffered_transcript()
    
    def process_buffered_transcript(self):
        """Process the buffered transcript through LLM"""
        if not self.transcript_buffer or self.is_processing:
            return
        
        self.is_processing = True
        
        # Get the complete transcript from buffer
        full_transcript = " ".join([item['text'] for item in self.transcript_buffer])
        
        # Clear buffer after processing
        self.transcript_buffer.clear()
        
        # Call LLM asynchronously
        self.llm_task = asyncio.run_coroutine_threadsafe(
            self.call_llm_async(full_transcript),
            self.loop
        )
    
    async def call_llm_async(self, text: str):
        """Asynchronous LLM call"""
        try:
            client = genai.Client()
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash", 
                contents=text
            )
            
            # Collect complete LLM response
            llm_response = ""
            for chunk in response:
                if chunk.text:
                    llm_response += chunk.text
                    await self.websocket.send_json({
                        "status": "llm_response",
                        "text": chunk.text,
                        "is_complete": False
                    })
                    print(chunk.text)
                    print("_" * 50)
            
            # Send completion signal
            await self.websocket.send_json({
                "status": "llm_response",
                "text": "",
                "is_complete": True
            })
            
            #Convert LLm response to speech using Murf.ai
            if llm_response.strip():
                print("Converting LLM response to speech...")
                await self.murf_service.synthesize_speech(llm_response)
            
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            await self.websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        finally:
            self.is_processing = False
     
    def on_terminated(self, client, event: TerminationEvent):
       print(f"Session terminated: {event.audio_duration_seconds} seconds processed")
       # Process any remaining buffered transcript
       if self.transcript_buffer:
            self.process_buffered_transcript()     
    
    def on_error(self,client, error: StreamingError):
       print(f"Error: {error}")
       
    def stream(self, audio_chunk: bytes):
        self.client.stream(audio_chunk)
        # Check for silence after each audio chunk
        self.check_silence_and_process()

    def close(self):
        # Process any remaining transcript before closing
        if self.transcript_buffer:
            self.process_buffered_transcript()
        self.client.disconnect(terminate=True)   
    



   