import streamlit as st
import socket

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to determine IP"

def show_sharing_info():
    """Display sharing information in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🌐 Share Dashboard")
        
        local_ip = get_local_ip()
        if local_ip != "Unable to determine IP":
            share_url = f"http://{local_ip}:8501"
            st.success(f"**Local Network URL:**")
            st.code(share_url)
            st.info("💡 Share this URL with team members on the same network")
        
        st.markdown("**For Public Access:**")
        st.markdown("1. Deploy to [Streamlit Cloud](https://share.streamlit.io)")
        st.markdown("2. Use [ngrok](https://ngrok.com) for instant public URL")
        st.markdown("3. See DEPLOYMENT_GUIDE.md for details")

# Add this to your main function
if __name__ == "__main__":
    show_sharing_info()
