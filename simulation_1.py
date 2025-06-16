import streamlit as st
import numpy as np
import simpy
import matplotlib.pyplot as plt

st.title("Simulacija reda čekanja")

num_patients = st.number_input("Unesi broj pacijenata u danu:", min_value=1, value=100)
service_time_min = st.number_input("Unesi prosječno trajanje pregleda (min):", min_value=1, value=60)
num_doctors = st.number_input("Unesi broj doktora:", min_value=1, value=5)
sim_hours = 24  

λ = num_patients / (sim_hours * 60)    
μ = 1 / service_time_min               

st.write(f"Izračunato: λ = {λ:.5f} dolazaka/min, μ = {μ:.5f} servisa/min")

def patient_arrival(env, doctors, lamda, mu, wait_times):
    while env.now < sim_hours * 60:
        inter_arrival = np.random.exponential(1 / lamda)
        yield env.timeout(inter_arrival)
        env.process(patient(env, doctors, mu, wait_times))

def patient(env, doctors, mu, wait_times):
    arrival_time = env.now
    with doctors.request() as req:
        yield req
        wait = env.now - arrival_time
        wait_times.append(wait)
        service_time = np.random.exponential(1 / mu)
        yield env.timeout(service_time)

def simulate_er(lambda_, mu, doctors_num, sim_hours=24):
    env = simpy.Environment()
    doctors = simpy.Resource(env, capacity=doctors_num)
    wait_times = []
    env.process(patient_arrival(env, doctors, lambda_, mu, wait_times))
    env.run(until=sim_hours * 60)
    return np.array(wait_times)

if st.button("Pokreni simulaciju"):
    wait_times = simulate_er(λ, μ, num_doctors, sim_hours)
    st.write(f"Broj pacijenata simuliran: {len(wait_times)}")
    st.write(f"Srednje vrijeme čekanja: {wait_times.mean():.2f} min")
    st.write(f"Medijan vremena čekanja: {np.median(wait_times):.2f} min")
    st.write(f"90. percentil čekanja: {np.percentile(wait_times, 90):.2f} min")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(wait_times, bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel("Vrijeme čekanja (min)")
    ax.set_ylabel("Broj pacijenata")
    ax.set_title(f"Distribucija vremena čekanja\n(λ={λ:.4f}, μ={μ:.4f}, doktora={num_doctors})")
    ax.grid(True)
    st.pyplot(fig)
