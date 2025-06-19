#!/usr/bin/env python3
"""
Quick tunnel status checker for F.R.E.D.
"""
import json
import os
from ollietec_theme import apply_theme

apply_theme()

def check_tunnel_status():
    """Check and display current tunnel status"""
    
    print("[NETWORK] Checking F.R.E.D. tunnel status...")
    
    # Check tunnel_info.json
    if os.path.exists('tunnel_info.json'):
        try:
            with open('tunnel_info.json', 'r') as f:
                tunnel_data = json.load(f)
                tunnel_url = tunnel_data.get('webrtc_server')
                
                if tunnel_url:
                    print("\n" + "="*80)
                    print("🌐 ACTIVE NGROK TUNNEL FOUND:")
                    print("="*80)
                    print(f"📡 URL: {tunnel_url}")
                    print(f"📱 Pi Command: python client.py --server {tunnel_url}")
                    print("="*80)
                    
                    # Test if tunnel is reachable
                    try:
                        import requests
                        response = requests.get(tunnel_url, timeout=5)
                        if response.status_code == 200:
                            print("[SUCCESS] Tunnel is ACTIVE and reachable!")
                        else:
                            print(f"[WARNING] Tunnel responds but may have issues (Status: {response.status_code})")
                    except Exception as e:
                        print(f"[ERROR] Tunnel may be down (Error: {e})")
                        
                else:
                    print("[ERROR] No tunnel URL found in tunnel_info.json")
                    
        except Exception as e:
            print(f"[ERROR] Error reading tunnel_info.json: {e}")
    else:
        print("[ERROR] No tunnel_info.json found")
        print("[INFO] Start F.R.E.D. with: python start_fred_with_webrtc.py")
    
    # Check if ngrok is running
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
        tunnels = response.json()
        
        print(f"\n🔧 Ngrok API Status: {len(tunnels.get('tunnels', []))} tunnel(s) running")
        for tunnel in tunnels.get('tunnels', []):
            public_url = tunnel.get('public_url', 'Unknown')
            local_addr = tunnel.get('config', {}).get('addr', 'Unknown')
            print(f"   📡 {public_url} → {local_addr}")
            
    except Exception:
        print("\n🔧 Ngrok API: Not accessible (ngrok may not be running)")

if __name__ == "__main__":
    check_tunnel_status() 