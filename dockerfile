# Set the base image to the official Rust nightly build
FROM rustlang/rust:nightly

# Install the required system libraries
RUN apt-get update && \
    apt-get install -y libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a new directory to copy the code into
RUN mkdir /app
WORKDIR /app

# Copy the Rust code into the container
COPY . .

# Build the Rust code
RUN cargo build --release

# Set the default command to run the binary
CMD ["/app/target/release/model-loader"]