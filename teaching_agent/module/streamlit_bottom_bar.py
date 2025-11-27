import streamlit as st


def bottom_bar(
    previous_page: str = None,
    previous_alias: str = None,
    next_page: str = None,
    next_alias: str = None,
) -> None:
    st.divider()
    with st.container(horizontal=True, key="bottom_bar"):
        if previous_page:
            if st.button(f"**← Previous:** {previous_alias}"):
                st.switch_page(previous_page)
        else:
            st.button(f"**← Previous:**", disabled=True)
        st.space("stretch")
        if next_page:
            if st.button(f"**Next:** {next_alias} **→**", type="primary"):
                st.switch_page(next_page)
        else:
            st.button("**Next:**", disabled=True)
