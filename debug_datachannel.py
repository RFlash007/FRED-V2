#!/usr/bin/env python3
"""
Minimal WebRTC Data Channel Test
Tests if aiortc data channels work between client and server
"""

import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription

async def test_datachannel():
    """Test data channel between two peer connections"""
    
    print("🧪 [TEST] Starting data channel test...")
    
    # Create two peer connections (simulating client and server)
    pc1 = RTCPeerConnection()
    pc2 = RTCPeerConnection()
    
    # Track data channel events
    dc1_opened = False
    dc2_opened = False
    
    # Client side (pc1) creates data channel
    dc1 = pc1.createDataChannel('test', ordered=True)
    print(f"🔧 [PC1] Data channel created: {dc1.label}, state: {dc1.readyState}")
    
    @dc1.on('open')
    def on_dc1_open():
        nonlocal dc1_opened
        dc1_opened = True
        print("✅ [PC1] Data channel opened!")
        dc1.send("Hello from PC1")
    
    @dc1.on('message')
    def on_dc1_message(message):
        print(f"📨 [PC1] Received: {message}")
    
    # Server side (pc2) handles incoming data channel
    @pc2.on('datachannel')
    def on_datachannel(channel):
        nonlocal dc2_opened
        print(f"🔧 [PC2] Data channel received: {channel.label}, state: {channel.readyState}")
        
        @channel.on('open')
        def on_dc2_open():
            nonlocal dc2_opened
            dc2_opened = True
            print("✅ [PC2] Data channel opened!")
            channel.send("Hello from PC2")
        
        @channel.on('message')
        def on_dc2_message(message):
            print(f"📨 [PC2] Received: {message}")
    
    # Connection state monitoring
    @pc1.on('connectionstatechange')
    async def on_pc1_state():
        print(f"🔗 [PC1] Connection: {pc1.connectionState}")
    
    @pc2.on('connectionstatechange') 
    async def on_pc2_state():
        print(f"🔗 [PC2] Connection: {pc2.connectionState}")
    
    @pc1.on('iceconnectionstatechange')
    async def on_pc1_ice():
        print(f"🧊 [PC1] ICE: {pc1.iceConnectionState}")
    
    @pc2.on('iceconnectionstatechange')
    async def on_pc2_ice():
        print(f"🧊 [PC2] ICE: {pc2.iceConnectionState}")
    
    try:
        # Create offer
        print("📤 [PC1] Creating offer...")
        offer = await pc1.createOffer()
        await pc1.setLocalDescription(offer)
        
        # Set remote description on server
        print("📥 [PC2] Setting remote description...")
        await pc2.setRemoteDescription(RTCSessionDescription(
            sdp=offer.sdp, type=offer.type
        ))
        
        # Create answer
        print("📤 [PC2] Creating answer...")
        answer = await pc2.createAnswer()
        await pc2.setLocalDescription(answer)
        
        # Set remote description on client
        print("📥 [PC1] Setting remote description...")
        await pc1.setRemoteDescription(RTCSessionDescription(
            sdp=answer.sdp, type=answer.type
        ))
        
        print("⏰ [TEST] Waiting for connections...")
        
        # Wait for connections to establish
        for i in range(30):  # 30 second timeout
            await asyncio.sleep(1)
            
            print(f"⏱️  [{i+1}s] PC1: {pc1.connectionState}/{pc1.iceConnectionState}, "
                  f"PC2: {pc2.connectionState}/{pc2.iceConnectionState}")
            print(f"     DC1: {dc1.readyState}, DC2 opened: {dc2_opened}")
            
            if dc1_opened and dc2_opened:
                print("🎉 [SUCCESS] Both data channels opened!")
                break
                
            if pc1.connectionState == "failed" or pc2.connectionState == "failed":
                print("❌ [FAILED] Connection failed")
                break
        else:
            print("⏰ [TIMEOUT] Data channels did not open within 30 seconds")
    
    finally:
        await pc1.close()
        await pc2.close()
        print("🧹 [CLEANUP] Connections closed")

if __name__ == "__main__":
    asyncio.run(test_datachannel()) 