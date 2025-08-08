#!/usr/bin/env python3
"""
Quick tunnel status checker for F.R.E.D.
"""
import json
import os

# Silent no-op print to remove dependency on ollietec_theme and ollie_print
def olliePrint(*args, **kwargs):
    return None

def check_tunnel_status():
    """Check and display current tunnel status"""
    
    olliePrint("[NETWORK] Checking F.R.E.D. tunnel status...")
    
    # Check tunnel_info.json
    if os.path.exists('tunnel_info.json'):
        try:
            with open('tunnel_info.json', 'r') as f:
                tunnel_data = json.load(f)
                tunnel_url = tunnel_data.get('webrtc_server')
                
                if tunnel_url:
                    olliePrint("\n" + "="*80)
                    olliePrint("🌐 ACTIVE NGROK TUNNEL FOUND:")
                    olliePrint("="*80)
                    olliePrint(f"📡 URL: {tunnel_url}")
                    olliePrint(f"📱 Pi Command: python client.py --server {tunnel_url}")
                    olliePrint("="*80)
                    
                    # Test if tunnel is reachable
                    try:
                        import requests
                        response = requests.get(tunnel_url, timeout=5)
                        if response.status_code == 200:
                            olliePrint("[SUCCESS] Tunnel is ACTIVE and reachable!")
                        else:
                            olliePrint(f"[WARNING] Tunnel responds but may have issues (Status: {response.status_code})")
                    except Exception as e:
                        olliePrint(f"[ERROR] Tunnel may be down (Error: {e})")
                        
                else:
                    olliePrint("[ERROR] No tunnel URL found in tunnel_info.json")
                    
        except Exception as e:
            olliePrint(f"[ERROR] Error reading tunnel_info.json: {e}")
    else:
        olliePrint("[ERROR] No tunnel_info.json found")
        olliePrint("[INFO] Start F.R.E.D. with: python start_fred_with_webrtc.py")
    
    # Check if ngrok is running
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
        tunnels = response.json()
        
        olliePrint(f"\n🔧 Ngrok API Status: {len(tunnels.get('tunnels', []))} tunnel(s) running")
        for tunnel in tunnels.get('tunnels', []):
            public_url = tunnel.get('public_url', 'Unknown')
            local_addr = tunnel.get('config', {}).get('addr', 'Unknown')
            olliePrint(f"   📡 {public_url} → {local_addr}")
            
    except Exception:
        olliePrint("\n🔧 Ngrok API: Not accessible (ngrok may not be running)")

if __name__ == "__main__":
    check_tunnel_status() 