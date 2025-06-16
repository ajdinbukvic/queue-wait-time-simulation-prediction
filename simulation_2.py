import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_mm_c(lambda_rate, mu_rate, c, simulation_time=10000):
    arrival_times = []
    time = 0.0
    while time < simulation_time:
        time += np.random.exponential(1 / lambda_rate)
        if time < simulation_time:
            arrival_times.append(time)

    num_patients = len(arrival_times)

    start_service_times = np.zeros(num_patients)
    end_service_times = np.zeros(num_patients)

    for i in range(num_patients):
        if i < c:
            start_service_times[i] = arrival_times[i]
        else:
            earliest_server_free_time = np.min(end_service_times[i - c:i])
            start_service_times[i] = max(arrival_times[i], earliest_server_free_time)
        service_time = np.random.exponential(1 / mu_rate)
        end_service_times[i] = start_service_times[i] + service_time

    wait_times_in_queue = start_service_times - arrival_times 
    total_times_in_system = end_service_times - arrival_times   
    average_Wq = np.mean(wait_times_in_queue)
    average_W = np.mean(total_times_in_system)

    average_Lq = lambda_rate * average_Wq
    average_L = lambda_rate * average_W

    rho = lambda_rate / (c * mu_rate)

    return {
        'Wq': average_Wq,
        'W': average_W,
        'Lq': average_Lq,
        'L': average_L,
        'rho': rho
    }

def main():
    st.title("Simulacija iskorištenosti sistema")

    patients_per_day = st.number_input("Broj pacijenata dnevno:", min_value=1, value=50)
    service_time = st.number_input("Prosječno trajanje pregleda (min):", min_value=1, value=80)
    c = st.number_input("Broj doktora (c):", min_value=1, value=4)

    lambda_per_minute = patients_per_day / (24 * 60)
    mu_per_minute = 1 / service_time

    st.write(f"Stepen dolazaka λ = {lambda_per_minute:.5f} pacijenata/min")
    st.write(f"Stepen servisa μ = {mu_per_minute:.5f} pacijenata/min po doktoru")

    if st.button("Pokreni simulaciju"):
        stats = simulate_mm_c(lambda_per_minute, mu_per_minute, c)
        st.write(f"Iskorištenost (ρ): {stats['rho']:.4f}")
        st.write(f"Prosječno čekanje u redu (Wq): {stats['Wq']:.2f} min")
        st.write(f"Ukupno vrijeme u sistemu (W): {stats['W']:.2f} min")
        st.write(f"Prosječan broj u redu (Lq): {stats['Lq']:.2f} pacijenata")
        st.write(f"Prosječan broj u sistemu (L): {stats['L']:.2f} pacijenata")

        c_range = range(1, max(c*2, 10))
        Wq_values = [simulate_mm_c(lambda_per_minute, mu_per_minute, ci)['Wq'] for ci in c_range]

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(c_range, Wq_values, marker='o')
        ax.set_title('Utjecaj broja doktora na prosječno čekanje u redu (Wq)')
        ax.set_xlabel('Broj doktora (c)')
        ax.set_ylabel('Prosječno čekanje u redu Wq [min]')
        ax.grid(True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
